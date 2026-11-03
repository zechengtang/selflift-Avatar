# Adapted for selflift-Avatar from Fish4AI/comfyui-SelfLift-Fish.
# Full-sequence H3 attention context is retained; Avatar masks stay on the full packed sequence.
"""H3 global token chunking, with Sol-Attn/SLA and Spark-H3 compatibility.

Unlike spatial cropping, this preserves the dense H3 attention operator. Q/K/V
projection, attention queries and the MLP are chunked; K/V stay complete on the
compute device and Q is staged on the CPU. This is inference-only and does not
reduce the quadratic attention work or the size of the complete K/V buffers.
With ComfyUI's BlockSparseAttention, retain its full-sequence sparse operator
and stream normalized projections into its int8 producer instead (no full
high-precision Q/K/V and no CPU Q staging). Its preceding attention backend
is retained on full-query native fallbacks. VSA and standalone attention
overrides are not adapted here. Spark retains its own block patch and attention
implementation, including optional chunked projection, with chunked norm/MLP.
"""

from functools import partial
import logging
import time
import inspect
from pathlib import Path
import sys
import weakref
from contextlib import nullcontext

from .h3_tiling import _is_spark_block_function, validate_spark_chunking

import torch

import comfy.model_base
import comfy.model_management
import comfy.patcher_extension
import comfy.quant_ops
import comfy.model_prefetch
try:
    import comfy_kitchen as ck
except ImportError:
    ck = None
from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention


def _segments_slice(segments, start, end):
    """Preserve scalar and per-token AdaLN row indices across chunk boundaries."""
    result = []
    for a, b, row in segments:
        lo, hi = max(a, start), min(b, end)
        if lo >= hi:
            continue
        if torch.is_tensor(row) and row.ndim:
            row = row[lo - a:hi - a]
        result.append((lo - start, hi - start, row))
    return result


def _normalized_qkv(attention, h, rope_freqs):
    """Use H3's own fused RMSNorm/RoPE path, including partial split-half RoPE."""
    s = h.shape[0]
    inner = attention.heads * attention.head_dim
    q, k, v = attention.qkv_proj(h).split(inner, dim=-1)
    v = v.view(s, attention.heads, attention.head_dim)
    if rope_freqs is None:
        q = attention.q_norm(q.view(s, attention.heads, attention.head_dim))
        k = attention.k_norm(k.view(s, attention.heads, attention.head_dim))
    else:
        q = q.view(1, s, attention.heads, attention.head_dim)
        k = k.view(1, s, attention.heads, attention.head_dim)
        qw = comfy.model_management.cast_to(attention.q_norm.weight, device=h.device)
        kw = comfy.model_management.cast_to(attention.k_norm.weight, device=h.device)
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k, rope_freqs, qw, kw, epsilon=attention.q_norm.eps,
            rot_dim=rope_freqs.shape[-3] * 2)
        q, k = q[0], k[0]
    return q, k, v


def chunked_block(block, x, t_emb, mod_segments, rope_freqs, transformer_options, chunk_size):
    """Dense full-context attention with bounded per-token intermediates.

    An installed attention override (for example TST or another H3 patch) is
    preserved by running the full-sequence attention operator once. Only the
    normalization and MLP rows are chunked in that case; query/key slicing is
    deliberately skipped because an override may depend on the whole sequence.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("selflift-Avatar global chunking is inference-only")
    if transformer_options.get("optimized_attention_override") is not None:
        return _native_attention_chunked_mlp(block, x, t_emb, mod_segments, rope_freqs,
                                             transformer_options, chunk_size)
    attention = block.attn
    preferred = getattr(attention, "comfy_attention", None)
    if getattr(preferred, "function", None) is not None:
        return _native_attention_chunked_mlp(block, x, t_emb, mod_segments, rope_freqs,
                                             transformer_options, chunk_size)
    s = x.shape[0]
    chunk_size = max(1, int(chunk_size))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    q_cpu = k_full = v_full = None
    # Build ALL keys/values from the unmodified layer input before updating any
    # residual rows. Chunking K/V attention independently would change softmax.
    for start in range(0, s, chunk_size):
        end = min(s, start + chunk_size)
        segments = _segments_slice(mod_segments, start, end)
        h = _mod_scale_shift(block.norm1(x[start:end]), shift_msa, scale_msa, segments)
        rope = rope_freqs[:, start:end] if rope_freqs is not None else None
        q, k, v = _normalized_qkv(attention, h, rope)
        if q_cpu is None:
            shape = (s, attention.heads, attention.head_dim)
            q_cpu = torch.empty(shape, dtype=q.dtype, device="cpu")
            k_full = torch.empty(shape, dtype=k.dtype, device=x.device)
            v_full = torch.empty(shape, dtype=v.dtype, device=x.device)
        q_cpu[start:end].copy_(q)
        k_full[start:end].copy_(k)
        v_full[start:end].copy_(v)
        del q, k, v, h
    # The key axis is NEVER cropped. Every query sees text, every reference,
    # complete audio, every video frame, and the complete spatial grid.
    for start in range(0, s, chunk_size):
        end = min(s, start + chunk_size)
        q = q_cpu[start:end].to(device=x.device)
        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        # Comfy's containers are single-use; wrap the shared tensors anew.
        k = AttentionTensorContainer(k_full.transpose(0, 1).unsqueeze(0))
        v = AttentionTensorContainer(v_full.transpose(0, 1).unsqueeze(0))
        out = optimized_attention(q, k, v, attention.heads, preferred_attention=preferred,
                                  mask=None, skip_reshape=True,
                                  transformer_options=transformer_options)
        out = attention.out_proj(out.squeeze(0))
        _mod_gate(x[start:end], gate_msa, out, _segments_slice(mod_segments, start, end))
        del q, k, v, out
    del q_cpu, k_full, v_full
    for start in range(0, s, chunk_size):
        end = min(s, start + chunk_size)
        segments = _segments_slice(mod_segments, start, end)
        h = _mod_scale_shift(block.norm2(x[start:end]), shift_mlp, scale_mlp, segments)
        out = block.mlp(h)
        _mod_gate(x[start:end], gate_mlp, out, segments)
        del h, out
    return x


def _is_native_sparse_function(fn, factory, leaf):
    code = getattr(fn, "__code__", None)
    return (code is not None and Path(code.co_filename).name == "nodes_sparse_attention.py"
            and fn.__qualname__ == f"{factory}.<locals>.{leaf}")


def _has_unknown_attention_override(options):
    override = options.get("optimized_attention_override")
    return override is not None and not _is_native_sparse_function(
        override, "make_attention_override", "override")


def _spark_specs(network, options):
    dit = options.get("patches_replace", {}).get("dit", {})
    specs = {}
    for i, block in enumerate(network.blocks):
        fn = dit.get(("double_block", i))
        if not _is_spark_block_function(fn):
            continue
        closure = inspect.getclosurevars(fn).nonlocals
        if closure.get("block") is not block or closure.get("layer") != i:
            raise ValueError("selflift-Avatar global: Spark patch does not match this H3 block")
        specs[i] = fn
    if specs and len(specs) != len(network.blocks):
        raise ValueError("selflift-Avatar global: Spark must patch every H3 block")
    if specs and options.get("optimized_attention_override") is not None:
        # Spark's sparse callback bypasses this hook; accepting it would silently
        # drop TST/other corrections. Dense backend selection belongs in Spark.
        raise ValueError("selflift-Avatar global: Spark with an external attention override is not adapted; select the dense backend in Spark")
    return specs


def _is_backend_attention_override(override):
    """True for ComfyUI's Model Attention Backend selector.

    That override only selects a dense backend; it is safe to drop for the
    full-sequence global fallback. TST/Motion Context overrides are not dropped.
    """
    qualname = getattr(override, "__qualname__", "") or ""
    return "set_model_optimized_attention" in qualname


def _sparse_specs(network, options, force_dense_on_unknown=False):
    """Recognize the installed native patch; reject unsafe patches before work.

    The native sparse hook may wrap a previously selected dense backend. That
    chain is safe only on full-query native fallbacks, never on sliced queries.
    With force_dense_on_unknown, a TST/other attention override is preserved by
    returning no sparse specs, so the caller uses the full-sequence attention
    operator with chunked MLP instead of silently ignoring the override.
    """
    dit = options.get("patches_replace", {}).get("dit", {})
    specs = {}
    for i, block in enumerate(network.blocks):
        fn = dit.get(("double_block", i))
        if fn is None:
            continue
        if _is_spark_block_function(fn):
            continue  # Validated separately by _spark_specs; never execute as Sol.
        if not _is_native_sparse_function(fn, "make_h3_block_patch", "block_patch"):
            raise ValueError("selflift-Avatar global: unsupported H3 block patch; native Model Sparse Attention (sol-attn/SLA) and Spark-H3 are supported")
        closure = inspect.getclosurevars(fn).nonlocals
        native = sys.modules.get(fn.__module__)
        patch = closure.get("patch")
        if (native is None or not isinstance(patch, getattr(native, "SparseAttnPatch", ()))
                or closure.get("block") is not block or closure.get("block_index") != i):
            raise ValueError("selflift-Avatar global: native sparse patch does not match this H3 block")
        if patch.vsa:
            raise ValueError("selflift-Avatar global: VSA is not adapted; this version supports sol-attn and SLA")
        specs[i] = (patch, native)
    override = options.get("optimized_attention_override")
    if override is not None:
        if not _is_native_sparse_function(override, "make_attention_override", "override"):
            if force_dense_on_unknown:
                # Keep the override and force the caller onto full-sequence
                # attention. tiled_model logs the one visible warning.
                return {}
            raise ValueError(
                "selflift-Avatar global: TST or an unknown attention override remains; "
                "use highres_tiling_mode=global or disable TST/other attention overrides"
            )
        closure = inspect.getclosurevars(override).nonlocals
        patches = {spec[0] for spec in specs.values()}
        if closure.get("patch") not in patches:
            raise ValueError("selflift-Avatar global: sparse attention override does not match the installed H3 block patches")
        previous = closure.get("previous")
        if previous is not None and not callable(previous):
            raise ValueError("selflift-Avatar global: the native sparse attention fallback must be callable")
    return specs


def _chunk_mlp(block, x, shift, scale, gate, segments, chunk_size):
    for start in range(0, x.shape[0], chunk_size):
        end = min(x.shape[0], start + chunk_size)
        seg = _segments_slice(segments, start, end)
        h = _mod_scale_shift(block.norm2(x[start:end]), shift, scale, seg)
        out = block.mlp(h)
        _mod_gate(x[start:end], gate, out, seg)
        del h, out


def _sparse_block(block, x, t_emb, mod_segments, rope_freqs, options, chunk_size, spec, block_index, ram_state=None, use_stream=False):
    """Native sparse semantics + fused/chunked norm, projection, residual, MLP."""
    if ck is None:
        raise RuntimeError("selflift-Avatar global sparse attention requires comfy_kitchen. Use highres_tiling_mode=global for the dense fallback.")
    if torch.is_grad_enabled():
        raise RuntimeError("selflift-Avatar global chunking is inference-only")
    patch, native = spec
    attn = block.attn
    n = x.shape[0]
    # Respect the SAME native eligibility, dense layer list, sigma window and
    # minimum sequence length; do not silently change the sparse schedule.
    if not native.h3_eligible(attn, x, rope_freqs, options, patch, block_index):
        preferred = getattr(attn, "comfy_attention", None)
        override = options.get("optimized_attention_override")
        # We may remove only this sparse hook when it would decline the call
        # and has no preceding backend. Otherwise keep the original attention
        # call with ALL query rows: backend quantization/statistics and any
        # row-indexed override must see the same shape as native H3.
        plain_dense = override is None
        if _is_native_sparse_function(override, "make_attention_override", "override"):
            closure = inspect.getclosurevars(override).nonlocals
            plain_dense = closure.get("patch") is patch and closure.get("previous") is None
        if (plain_dense and patch.dense_reason(options, n, block_index) is not None
                and getattr(preferred, "function", None) is None):
            dense_options = options.copy()
            dense_options.pop("optimized_attention_override", None)
            return chunked_block(block, x, t_emb, mod_segments, rope_freqs, dense_options, chunk_size)
        # Unusual dtype/head/backend: keep the native full-attention fallback,
        # which may still accelerate via its generic attention override.
        return _native_attention_chunked_mlp(block, x, t_emb, mod_segments, rope_freqs, options, chunk_size)

    if use_stream:
        from . import h3_stream_attention
        return h3_stream_attention.sparse_block(block, x, t_emb, mod_segments, rope_freqs,
                                                options, chunk_size, spec, block_index)
    if ram_state is not None:
        from . import h3_ram_attention
        return h3_ram_attention.sparse_block(block, x, t_emb, mod_segments, rope_freqs,
                                             options, chunk_size, spec, block_index, ram_state)

    # Producer chunks must begin at 64-row boundaries. UI lengths are already
    # multiples of 128; normalize direct API calls as well.
    chunk_size = max(64, int(chunk_size) // 64 * 64)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    heads, dim = attn.heads, attn.head_dim
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
    key = (block_index, n, tuple(options.get("uuids", ())))
    with comfy.model_prefetch.pause_malloc_graph():
        pooled = patch.pooled.get(key)
        first = pooled is None
        if first:
            pooled = (torch.empty((heads, dim), dtype=torch.float32, device=x.device),
                      torch.empty((heads, dim), dtype=torch.float32, device=x.device))
    sink, sink_q = patch.sinks(options, n)

    def chunks():
        # The bootstrap calls this factory twice. x must stay untouched until
        # BOTH passes and the full sparse attention finish.
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            seg = _segments_slice(mod_segments, start, end)
            h = _mod_scale_shift(block.norm1(x[start:end]), shift_msa, scale_msa, seg)
            yield attn.qkv_proj(h)

    out, kmean, vscale = ck.sol_attn_chunked(
        chunks, n, heads, rope_freqs, (qw, kw),
        kmean=None if first else pooled[0], vscale=None if first else pooled[1],
        tau=patch.tau, topk_ratio=patch.topk_ratio, token_aug=patch.extra_tokens,
        sink_blocks=list(sink), sink_q=list(sink_q), rope_eps=attn.q_norm.eps)
    pooled[0].copy_(kmean)
    pooled[1].copy_(vscale)
    patch.pooled[key] = pooled
    out = out.view(n, heads * dim)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        projected = attn.out_proj(out[start:end])
        _mod_gate(x[start:end], gate_msa, projected, _segments_slice(mod_segments, start, end))
        del projected
    del out, kmean, vscale
    _chunk_mlp(block, x, shift_mlp, scale_mlp, gate_mlp, mod_segments, chunk_size)
    log_key = ("selflift_global_v21", n, chunk_size)
    if log_key not in patch._logged:
        patch._logged.add(log_key)
        logging.info("[selflift-Avatar global sparse v2.1] tokens=%d chunk=%d mode=%s sinks=%s/%s; full-layout int8 producer",
                     n, chunk_size, "SLA" if patch.topk_ratio else "sol-attn", sink, sink_q)
    return x


def _native_attention_chunked_mlp(block, x, t_emb, segments, rope, options, chunk_size, attention=None):
    shift, scale, gate, mshift, mscale, mgate = block.adaln_proj(t_emb)
    h = torch.empty_like(x)
    for start in range(0, x.shape[0], chunk_size):
        end = min(x.shape[0], start + chunk_size)
        hc = _mod_scale_shift(block.norm1(x[start:end]), shift, scale, _segments_slice(segments, start, end))
        h[start:end].copy_(hc)
        del hc
    out = (block.attn if attention is None else attention)(h, rope_freqs=rope, transformer_options=options)
    del h
    _mod_gate(x, gate, out, segments)
    del out
    _chunk_mlp(block, x, mshift, mscale, mgate, segments, chunk_size)
    return x


def _block_replace(block, chunk_size, spec, index, use_stream, ram_state, args, extra):
    if spec is not None:
        return {"img": _sparse_block(block, args["img"], args["t_emb"], args["mod_segments"],
                                      args["rope_freqs"], args["transformer_options"], chunk_size, spec, index, ram_state, use_stream)}
    return {"img": chunked_block(block, args["img"], args["t_emb"], args["mod_segments"],
                                  args["rope_freqs"], args["transformer_options"], chunk_size)}


def _spark_drop_k_block(block, args, run_group, chunk_size, use_stream):
    """Sol-style block producer with an independent Spark reblock/core entry."""
    from .h3_prepared_weights import prepared_block
    x = args['img']
    scope = prepared_block(block, x.device, x.dtype) if use_stream else nullcontext()
    with scope:
        return _spark_drop_k_body(block, args, run_group, chunk_size, use_stream)


def _spark_drop_k_body(block, args, run_group, chunk_size, use_stream):
    from .h3_stream_attention import cast_projection, select_weight_rows
    x, segments = args['img'], args['mod_segments']
    options, rope = args['transformer_options'], args['rope_freqs']
    attn = block.attn
    n, heads, dim = x.shape[0], int(attn.heads), int(attn.head_dim)
    chunk_size = max(64, int(chunk_size) // 64 * 64)
    shift, scale, gate, mshift, mscale, mgate = block.adaln_proj(args['t_emb'])
    mod_rows = torch.empty(n, dtype=torch.long, device=x.device)
    for start, end, row in segments:
        if torch.is_tensor(row):
            row = row.to(device=x.device, dtype=torch.long)
        mod_rows[start:end] = row
    # Honor KJ's head-count split once; otherwise use Sol's 8-head stream groups.
    if 'minimax_head_chunks' in options:
        count = max(1, min(heads, int(options['minimax_head_chunks'])))
        sizes = [heads // count + (i < heads % count) for i in range(count)]
    else:
        group = min(heads, 8) if use_stream else heads
        sizes = [min(group, heads - a) for a in range(0, heads, group)]
    output = torch.empty(n, heads * dim, dtype=x.dtype, device=x.device)
    smoothing = getattr(attn.qkv_proj, 'pre_quant_scale', None)
    if smoothing is not None:
        smoothing = comfy.model_management.cast_to_device(smoothing, x.device, x.dtype)
    with cast_projection(attn.qkv_proj, x.device, x.dtype) as (full_weight, full_bias):
        head_start = 0
        for size in sizes:
            head_end = head_start + size
            cached_kind, weight, bias = None, None, None

            def project(kind, indices):
                nonlocal cached_kind, weight, bias
                if cached_kind != kind:
                    weight = bias = None
                    parts = {'q': (0,), 'k': (1,), 'kv': (1, 2), 'qkv': (0, 1, 2)}[kind]
                    channels = torch.arange(head_start * dim, head_end * dim, device=x.device)
                    spans = [channels + p * heads * dim for p in parts]
                    # Match Spark's projection GEMM geometry: concatenating the
                    # weight rows into a larger GEMM can alter BF16 rounding and
                    # consequently content-derived reblock/routing decisions.
                    if kind == 'qkv' and size == heads:
                        spans = [torch.cat(spans)]
                    weight = [select_weight_rows(full_weight, rows) for rows in spans]
                    bias = [full_bias.index_select(0, rows) if full_bias is not None else None
                            for rows in spans]
                    cached_kind = kind
                pieces = []
                for selected in indices.split(chunk_size):
                    h = block.norm1(x.index_select(0, selected))
                    h = _mod_scale_shift(h, shift, scale,
                        [(0, selected.numel(), mod_rows.index_select(0, selected))])
                    if smoothing is not None:
                        h = h * smoothing
                    projected = [torch.nn.functional.linear(h, w, b) for w, b in zip(weight, bias)]
                    pieces.append(projected[0] if len(projected) == 1 else torch.cat(projected, dim=-1))
                return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

            # Finish reblock and release K2 before replaying QKV into the shared
            # Sol producer. run_group returns the original H3 token order.
            out = run_group(project, n, rope, options, head_start, head_end, chunk_size)
            output[:, head_start * dim:head_end * dim].copy_(out)
            del out
            weight = bias = None
            head_start = head_end
    # All head groups finish replaying the unmodified x before residual writes.
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        out = attn.out_proj(output[start:end])
        _mod_gate(x[start:end], gate, out, _segments_slice(segments, start, end))
    del output, out
    _chunk_mlp(block, x, mshift, mscale, mgate, segments, chunk_size)
    return x


def _spark_block_replace(block, spark_patch, chunk_size, args, extra, use_stream=False):
    """Keep Spark's schedule; use the SelfLift producer for drop_k.

    Legacy/direct/keep_k callbacks retain the full-normalized-hidden path.
    Neither path adds Sol's CPU QKV staging.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("selflift-Avatar global chunking is inference-only")

    def original_block(block_args):
        attention = block_args.get('attention')
        drop_k = getattr(attention, 'selflift_drop_k', None)
        if callable(drop_k):
            return {'img': _spark_drop_k_block(block, block_args, drop_k, chunk_size, use_stream)}
        return {"img": _native_attention_chunked_mlp(
            block, block_args["img"], block_args["t_emb"], block_args["mod_segments"],
            block_args["rope_freqs"], block_args["transformer_options"], chunk_size,
            attention=block_args.get("attention"))}

    return spark_patch(args, {**extra, "original_block": original_block})


def _forward(executor, streams, timestep, context, transformer_options, minimax_payload=None,
             *, chunk_size=2048, use_ram=False, use_stream=False, base_model_ref=None, weight_cache_gb=4., **kwargs):
    network = executor.class_obj
    if network is None or not hasattr(network, "blocks"):
        raise ValueError("selflift-Avatar global chunking requires ComfyUI's native H3 model")
    options = comfy.patcher_extension.copy_nested_dicts(transformer_options)
    spark_specs = _spark_specs(network, options)
    specs = _sparse_specs(network, options, force_dense_on_unknown=True)
    if _is_backend_attention_override(options.get("optimized_attention_override")):
        # A backend selector is a performance choice, not a model semantic.
        # Full-sequence fallback uses the default attention path so a backend
        # that only supports short/tiled sequences cannot produce garbage.
        options.pop("optimized_attention_override", None)
    ram_state = {} if use_ram else None
    dit = options.setdefault("patches_replace", {}).setdefault("dit", {})
    for i, block in enumerate(network.blocks):
        key = ("double_block", i)
        if i in spark_specs:
            dit[key] = partial(_spark_block_replace, block, spark_specs[i], chunk_size, use_stream=use_stream)
        else:
            dit[key] = partial(_block_replace, block, chunk_size, specs.get(i), i, use_stream, ram_state)
    budget=nullcontext()
    output_scope=nullcontext()
    if use_stream:
        from .h3_prepared_weights import weight_cache_budget
        from .h3_output_chunking import output_chunking,trim_workspace_cache
        trim_workspace_cache(streams[0].device)
        base=base_model_ref() if base_model_ref is not None else None
        budget=weight_cache_budget(getattr(base,"current_patcher",None),weight_cache_gb)
        output_scope=output_chunking(network,chunk_size)
    try:
        with budget,output_scope:
            started=time.perf_counter()
            result=executor(streams, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
            if use_stream and streams[0].device.type=='cuda':
                logging.info('[selflift-Avatar global stream forward] wall=%.3fs (no forced CUDA sync); allocated=%.2f GiB reserved=%.2f GiB',time.perf_counter()-started,torch.cuda.memory_allocated(streams[0].device)/2**30,torch.cuda.memory_reserved(streams[0].device)/2**30)
            return result
    finally:
        if ram_state is not None:
            ram_state.clear()


def tiled_model(model, chunk_size=2048, use_ram=False, use_stream=False, weight_cache_gb=4.):
    if not isinstance(model.model, comfy.model_base.MiniMaxH3):
        raise ValueError("selflift-Avatar global chunking requires a MiniMax H3 model")
    if chunk_size < 1:
        raise ValueError("selflift-Avatar global chunk size must be positive")
    if weight_cache_gb<0:
        raise ValueError("selflift-Avatar weight-cache budget must be nonnegative")
    # Fail before spending the low-resolution prefix on unsupported patches.
    network = model.get_model_object("diffusion_model")
    if network is None or not hasattr(network, "blocks"):
        raise ValueError("selflift-Avatar global chunking requires ComfyUI's native H3 model with transformer blocks")
    options = model.model_options.get("transformer_options", {})
    spark_specs = _spark_specs(network, options)
    validate_spark_chunking(model)
    unknown_override = _has_unknown_attention_override(options)
    specs = _sparse_specs(network, options, force_dense_on_unknown=True)
    if use_ram and use_stream:
        raise ValueError("selflift-Avatar: choose global_ram or global_stream, not both")
    backend_override = _is_backend_attention_override(options.get("optimized_attention_override"))
    fallback_kind = ("attention_backend" if backend_override else "attention_override") if unknown_override else None
    if unknown_override:
        override = options.get("optimized_attention_override")
        override_name = (getattr(override, "__qualname__", None)
                         or getattr(override, "__name__", None)
                         or type(override).__name__)
        if backend_override:
            logging.warning(
                "[selflift-Avatar global] dense attention backend override detected (%s); "
                "using the default full-sequence attention path so long-sequence fallback is stable. "
                "Disable the Model Attention Backend node to avoid this fallback.",
                override_name,
            )
        elif use_ram or use_stream:
            logging.warning(
                "[selflift-Avatar global] attention override detected (%s); disabling head/GPU "
                "streaming; Falling back to the full-sequence global path with chunked MLP. "
                "Expect higher VRAM use; disable TST/other attention overrides for global_stream/global_ram.",
                override_name,
            )
        if use_ram or use_stream:
            use_ram = use_stream = False
    if (use_ram or use_stream) and not spark_specs and (not specs or len(specs) != len(network.blocks)):
        raise ValueError("selflift-Avatar global_ram/global_stream requires native Model Sparse Attention (sol-attn/SLA) or Spark-H3 on every H3 block")
    if use_stream and not spark_specs:
        from . import h3_stream_attention
        for block in network.blocks:
            h3_stream_attention.validate_projection(block.attn.qkv_proj)
    patched = model.clone()
    if spark_specs:
        drop_k = all(callable(getattr(inspect.getclosurevars(fn).nonlocals.get('attention'),
                                             'selflift_drop_k', None)) for fn in spark_specs.values())
        if drop_k:
            from .h3_stream_attention import validate_projection
            for block in network.blocks:
                validate_projection(block.attn.qkv_proj)
        patched._selflift_global_attention = "spark_drop_k" if drop_k else "spark"
        logging.info("[selflift-Avatar Spark] %s; full source hidden state retained; no CPU QKV staging",
                     "independent reblock + SelfLift QKV producer, head groups, output projection and MLP"
                     if drop_k else "Spark callback + SelfLift norm/MLP chunks")
    if fallback_kind:
        patched._selflift_global_fallback = fallback_kind
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 "selflift_global_chunking", partial(_forward, chunk_size=chunk_size, use_ram=use_ram, use_stream=use_stream,
                                                                    base_model_ref=weakref.ref(model.model),weight_cache_gb=weight_cache_gb))
    # Retain native full-shape memory planning. The legacy spatial-tile memory
    # estimate is unsafe here: full hidden-state and K/V buffers still exist.
    logging.info("[selflift-Avatar global chunking v2.1] chunk=%d; native sparse blocks=%d/%d; Spark blocks=%d/%d; native memory planning",
                 chunk_size, len(specs), len(network.blocks), len(spark_specs), len(network.blocks))
    if use_ram and not spark_specs:
        logging.info("[selflift-Avatar global RAM v3] enabled; 8-head groups, complete context, BF16 host staging; dense fallback unchanged")
    if use_stream and spark_specs:
        logging.info("[selflift-Avatar Spark stream] input/output chunking and weight-cache budget retained; cache cap=%.2f GiB", weight_cache_gb)
    elif use_stream:
        logging.info("[selflift-Avatar global stream] enabled; chunked input/output projections; GPU-only 8-head groups; block-local prepared LoRA; weight-cache cap=%.2f GiB (0=auto); native sparse and backend fallback retained",weight_cache_gb)
    return patched
