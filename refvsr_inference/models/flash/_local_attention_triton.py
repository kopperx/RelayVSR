"""Fused query-centred local attention for inference (forward only)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_relative(x, COSINE, SINE, lag, TD: tl.constexpr, INVERSE: tl.constexpr = False):
    """Rotate adjacent temporal channel pairs of an [N,D] tile."""
    d = tl.arange(0, x.shape[1])
    cosine = tl.load(COSINE + lag * TD + d, d < TD, 1).to(tl.float32)
    sine = tl.load(SINE + lag * TD + d, d < TD, 0).to(tl.float32)
    if INVERSE:
        sine = -sine
    paired = tl.gather(x, tl.broadcast_to((d ^ 1)[None, :], x.shape), axis=1)
    sign = tl.where(d % 2 == 0, -1.0, 1.0)
    # Match eager RoPE's multiply-then-add rounding for BF16/FP16 keys.
    direct = (x.to(tl.float32) * cosine[None, :]).to(x.dtype).to(tl.float32)
    cross = (paired.to(tl.float32) * sine[None, :]).to(x.dtype).to(tl.float32)
    return (direct + cross * sign[None, :]).to(x.dtype)


@triton.jit
def _forward(
    Q,
    K,
    V,
    OUT,
    LSE,
    PLANE_MASK,
    REL_COS,
    REL_SIN,
    TD: tl.constexpr,
    HAS_PLANE_MASK: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    WHS: tl.constexpr,
    WWS: tl.constexpr,
    COLS: tl.constexpr,
    QF: tl.constexpr,
    NF: tl.constexpr,
    CP: tl.constexpr,
    TW: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    n: tl.constexpr = HEIGHT * WIDTH
    nf: tl.constexpr = NF
    tile_y, tile_x, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qt = tile_y // triton.cdiv(HEIGHT, BH)
    tile_y = tile_y % triton.cdiv(HEIGHT, BH)
    batch, head = bh // HQ, bh % HQ
    kv_head = head // (HQ // HK)
    row = tl.arange(0, BM)
    qy, qx = tile_y * BH + row // BW, tile_x * BW + row % BW
    qi = qt * n + qy * WIDTH + qx
    valid_q = (qy < HEIGHT) & (qx < WIDTH)
    d = tl.arange(0, D)
    q = tl.load(
        Q + ((batch * HQ + head) * QF * n + qi[:, None]) * D + d[None, :],
        valid_q[:, None],
        0,
    )
    acc = tl.full((BM, D), 0, tl.float32)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    scale: tl.constexpr = D**-0.5 * 1.4426950408889634
    for source in tl.static_range(len(WHS)):
        if CAUSAL and source >= CP:
            plane = CP + qt - TW + 1 + source - CP
            valid_plane = plane >= CP
        else:
            plane = source
            valid_plane = True
        if HAS_PLANE_MASK:
            # Dynamo's Triton mutation analysis may represent bool pointers as
            # uint8. Load masks must remain i1 in both tracing and execution.
            valid_plane = valid_plane & tl.load(
                PLANE_MASK + batch * NF + plane, mask=valid_plane, other=0
            ).to(tl.int1)
        wh = WHS[source]
        ww = WWS[source]
        cols = COLS
        rows = wh + BH - 1
        top, left = tile_y * BH - wh // 2, tile_x * BW - ww // 2
        for offset in range(tl.cdiv(rows * cols, BN)):
            p = offset * BN + tl.arange(0, BN)
            ky, kx = top + p // cols, left + p % cols
            valid_k = (p < rows * cols) & (ky >= 0) & (ky < HEIGHT) & (kx >= 0) & (kx < WIDTH)
            valid_k = valid_k & valid_plane
            ki = plane * n + ky * WIDTH + kx
            k = tl.load(
                K + ((batch * HK + kv_head) * nf * n + ki[None, :]) * D + d[:, None],
                valid_k[None, :],
                0,
            )
            if TD and source >= CP:
                k = tl.trans(_rotate_relative(tl.trans(k), REL_COS, REL_SIN, qt + CP - plane, TD))
            scores = tl.dot(q, k).to(tl.float32) * scale
            allowed = (
                valid_k[None, :]
                & (ky[None, :] >= qy[:, None] - wh // 2)
                & (ky[None, :] < qy[:, None] + wh // 2)
                & (kx[None, :] >= qx[:, None] - ww // 2)
                & (kx[None, :] < qx[:, None] + ww // 2)
            )
            scores = tl.where(allowed, scores, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(scores, 1))
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            probabilities = tl.exp2(scores - safe_max[:, None])
            correction = tl.exp2(maximum - safe_max)
            v = tl.load(
                V + ((batch * HK + kv_head) * nf * n + ki[:, None]) * D + d[None, :],
                valid_k[:, None],
                0,
            )
            acc = acc * correction[:, None] + tl.dot(probabilities.to(v.dtype), v)
            denominator = denominator * correction + tl.sum(probabilities, 1)
            maximum = new_max
    if HAS_PLANE_MASK:
        result = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    else:
        result = acc / denominator[:, None]
    tl.store(
        OUT + ((batch * HQ + head) * QF * n + qi[:, None]) * D + d[None, :],
        result,
        valid_q[:, None],
    )
    tl.store(LSE + (batch * HQ + head) * QF * n + qi, maximum + tl.log2(denominator), valid_q)


@torch.no_grad()
def triton_local_attention(
    query,
    key,
    value,
    height,
    width,
    windows,
    query_frames=1,
    condition_planes=None,
    temporal_window=1,
    kv_plane_mask=None,
    relative_rope=None,
):
    with torch.cuda.device(query.device):
        return _run(
            query,
            key,
            value,
            height,
            width,
            windows,
            query_frames,
            condition_planes,
            temporal_window,
            kv_plane_mask,
            relative_rope,
        )


def _run(
    query,
    key,
    value,
    height,
    width,
    windows,
    query_frames,
    condition_planes,
    temporal_window,
    kv_plane_mask,
    relative_rope,
):
    q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
    plane_mask = kv_plane_mask.contiguous() if kv_plane_mask is not None else None
    relative_cos, relative_sin = (
        tuple(table.contiguous() for table in relative_rope)
        if relative_rope is not None
        else (None, None)
    )
    temporal_dim = relative_cos.shape[-1] if relative_cos is not None else 0
    b, hq, tokens, d = q.shape
    hk = k.shape[1]
    output = torch.empty_like(q)
    lse = torch.empty((b, hq, tokens), device=q.device, dtype=torch.float32)
    causal = condition_planes is not None
    cp = condition_planes if causal else 0
    selected_windows = windows[:cp] + (windows[cp],) * temporal_window if causal else windows
    whs, wws = (
        tuple(w[0] for w in selected_windows),
        tuple(w[1] for w in selected_windows),
    )
    bh, bw, bn, warps, stages = 4, 16, 64, 4, 2
    grid = (query_frames * triton.cdiv(height, bh), triton.cdiv(width, bw), b * hq)
    _forward[grid](
        q,
        k,
        v,
        output,
        lse,
        plane_mask,
        relative_cos,
        relative_sin,
        temporal_dim,
        plane_mask is not None,
        height,
        width,
        hq,
        hk,
        d,
        whs,
        wws,
        triton.next_power_of_2(max(wws) + bw - 1),
        query_frames,
        len(windows),
        cp,
        temporal_window,
        causal,
        bh,
        bw,
        bh * bw,
        bn,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=temporal_dim == 0,
    )
    return output
