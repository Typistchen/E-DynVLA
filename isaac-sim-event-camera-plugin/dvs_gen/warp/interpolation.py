"""
interpolation.py
================
Pluggable frame-interpolation strategies that accelerate the slow sim->RGB step.

Design
------
Strategies are registered by name so new interpolation methods can be added
without touching the simulation loop:

    from dvs_gen.warp import build_interpolator

    interp = build_interpolator("motion_vector", frames_per_keyframe=8)

    # ... in the main loop, only on keyframe steps (render_interval == K) ...
    mid = interp(rgb_keyframe, motion_vectors=mv)   # list of length K-1
    proc(rgb_keyframe, t_key)                        # the real frame
    for i, frame in enumerate(mid, start=1):
        proc(frame, t_key + i * dt)                  # synthesized frames

Add a method by subclassing ``FrameInterpolator`` and decorating it with
``@register("your_name")``.

Notes / assumptions
-------------------
* Tensors follow the camera-output layout ``(N, H, W, C)`` for RGB and
  ``(N, H, W, 2)`` for ``motion_vectors``. MEASURED units: the latter is
  per-pixel screen displacement in PIXELS (a fast object hits ~5 px/frame),
  NOT the normalized value the docs implied.
* The camera is assumed to be rendered once per keyframe gap
  (``render_interval == K``) so each motion-vector field spans the whole gap;
  the intermediate at fraction ``f = i/K`` moves content by ``f`` of it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────
# Registry  (lets callers pick a strategy by name at runtime)
# ──────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type["FrameInterpolator"]] = {}


def register(name: str):
    def _deco(cls: type["FrameInterpolator"]) -> type["FrameInterpolator"]:
        _REGISTRY[name] = cls
        return cls
    return _deco


def available_interpolators() -> list[str]:
    return sorted(_REGISTRY)


def build_interpolator(name: str, **kwargs) -> "FrameInterpolator":
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown interpolator '{name}'. Available: {available_interpolators()}"
        )
    return _REGISTRY[name](**kwargs)


# ──────────────────────────────────────────────────────────────
# Base strategy
# ──────────────────────────────────────────────────────────────
class FrameInterpolator(ABC):
    """Produce the ``K-1`` frames that fall strictly between one keyframe and
    the next. The caller still feeds the real keyframe itself."""

    def __init__(self, frames_per_keyframe: int = 8):
        assert frames_per_keyframe >= 1, "frames_per_keyframe (K) must be >= 1"
        self.K = int(frames_per_keyframe)

    @property
    def num_intermediate(self) -> int:
        return self.K - 1

    @abstractmethod
    def __call__(self, rgb: torch.Tensor, **aux) -> list[torch.Tensor]:
        """Return ``num_intermediate`` synthesized frames at time fractions
        ``1/K, 2/K, ..., (K-1)/K`` between this keyframe and the next.
        Each frame keeps the input ``(N, H, W, C)`` layout and dtype."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────
# "none" — baseline / A-B control (no synthesis)
# ──────────────────────────────────────────────────────────────
@register("none")
class NoInterpolation(FrameInterpolator):
    """Emit nothing between keyframes. Equivalent to simply rendering at the
    lower keyframe rate. Useful as the control when measuring whether warping
    actually improves the event stream."""

    def __call__(self, rgb: torch.Tensor, **aux) -> list[torch.Tensor]:
        return []


# ──────────────────────────────────────────────────────────────
# Motion-vector strategy — INTERFACE ONLY (implementation removed)
# ──────────────────────────────────────────────────────────────
@register("backward_splat")
class MotionVectorBackwardSplat(FrameInterpolator):
    """Backward motion-vector warp from the NEXT keyframe — interface stub.

    Contract (what an implementation must honour):
      * needs ``rgb_next`` (the next keyframe B) and ``motion_vectors_next``
        (mv_B, the true A->B correspondence) in ``aux``;
      * ``motion_vectors`` is scaled by the caller to span the whole K-gap;
      * synthesize each intermediate at fraction ``f = i/K`` by moving B's
        pixels back along their own motion to ``p - (1-f)*mv_B``;
      * return ``num_intermediate`` frames in ``(N, H, W, C)`` layout/dtype.

    The implementation has been removed; only the registered interface remains.
    """

    def __init__(self, frames_per_keyframe: int = 8, flow_sign: float = -1.0,
                 splat_hw: float = 1.0, beta: float = 12.0):
        super().__init__(frames_per_keyframe)
        self.flow_sign = float(flow_sign)
        self.splat_hw = float(splat_hw)
        self.beta = float(beta)

    def __call__(self, rgb: torch.Tensor, *, motion_vectors=None, rgb_next=None,
                 motion_vectors_next=None, **aux) -> list[torch.Tensor]:
        raise NotImplementedError("backward_splat implementation removed; interface only")


# ──────────────────────────────────────────────────────────────
# Dense-mv bidirectional warp  (IMPLEMENTED)
# ──────────────────────────────────────────────────────────────
def _splat(img, dx, dy, imp=None, hw=1.0, beta=12.0):
    """Scatter each source pixel of img (C,H,W) to (x+dx, y+dy). Collisions
    resolved by softmax splatting on importance ``imp`` (foreground wins).
    Returns (splatted (C,H,W), hole_mask (H,W) where nothing landed)."""
    import math
    C, H, W = img.shape
    dev = img.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    tx = xs.float() + dx
    ty = ys.float() + dy
    x0 = torch.floor(tx).long(); y0 = torch.floor(ty).long()
    if imp is None:
        imp_mult = torch.ones(H * W, device=dev)
    else:
        imp_mult = torch.exp(beta * (imp / (imp.max() + 1e-6))).reshape(-1)
    color = torch.zeros(C, H * W, device=dev)
    weight = torch.zeros(H * W, device=dev)
    src = img.reshape(C, -1)
    r = int(math.ceil(hw))
    for ox in range(1 - r, 1 + r):
        for oy in range(1 - r, 1 + r):
            xx = x0 + ox; yy = y0 + oy
            wgt = (1 - (tx - xx.float()).abs() / hw).clamp(min=0) * \
                  (1 - (ty - yy.float()).abs() / hw).clamp(min=0)
            valid = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            wflat = (wgt * valid.float()).reshape(-1) * imp_mult
            idx = (yy * W + xx).reshape(-1).clamp(0, H * W - 1)
            color.index_add_(1, idx, src * wflat.unsqueeze(0))
            weight.index_add_(0, idx, wflat)
    out = color / weight.clamp(min=1e-4).unsqueeze(0)
    holes = (weight < 1e-4).reshape(H, W)
    return out.reshape(C, H, W), holes


def _splat_batch(img, dx, dy, imp, hw=1.0, beta=12.0):
    """Batched softmax-splat: ``img`` (M,C,H,W); ``dx``,``dy``,``imp`` (M,H,W).
    All M independent splats run in ONE index_add (the batch index is folded into
    the scatter target). Returns out (M,C,H,W), holes (M,H,W)."""
    import math
    M, C, H, W = img.shape
    dev = img.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    xs = xs.float(); ys = ys.float()
    tx = xs + dx                                       # (M,H,W)
    ty = ys + dy
    x0 = torch.floor(tx).long(); y0 = torch.floor(ty).long()
    imp_max = imp.flatten(1).amax(1).clamp(min=1e-6).view(M, 1, 1)
    imp_mult = torch.exp(beta * (imp / imp_max))       # (M,H,W) — foreground/fast wins
    HW = H * W
    color = torch.zeros(C, M * HW, device=dev)
    weight = torch.zeros(M * HW, device=dev)
    src = img.permute(1, 0, 2, 3).reshape(C, M * HW)
    boff = (torch.arange(M, device=dev) * HW).view(M, 1, 1)     # per-batch index offset
    r = int(math.ceil(hw))
    for ox in range(1 - r, 1 + r):
        for oy in range(1 - r, 1 + r):
            xx = x0 + ox; yy = y0 + oy
            wgt = (1 - (tx - xx.float()).abs() / hw).clamp(min=0) * \
                  (1 - (ty - yy.float()).abs() / hw).clamp(min=0)
            valid = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            wflat = (wgt * valid.float() * imp_mult).reshape(-1)
            idx = ((yy * W + xx).clamp(0, HW - 1) + boff).reshape(-1)
            color.index_add_(1, idx, src * wflat.unsqueeze(0))
            weight.index_add_(0, idx, wflat)
    out = (color / weight.clamp(min=1e-4)).reshape(C, M, H, W).permute(1, 0, 2, 3)
    holes = (weight < 1e-4).reshape(M, H, W)
    return out, holes


def dilate_mv(mv, radius=1):
    """Grow the larger-magnitude motion over its neighbourhood so a moving
    object's motion vector covers its own anti-aliased silhouette fringe.

    The renderer anti-aliases the colour edge (the 1-px fringe is object-tinted)
    but classifies the motion vector per pixel, so that fringe is tagged with the
    static background's zero motion. Splatting then leaves the fringe behind at
    the object's keyframe position — a faint ghost outline that fires spurious
    events. Extending the object mv outward by ``radius`` px makes the fringe
    travel with the object and removes the ghost. ``mv``: ``(...,H,W,2)``.
    """
    if radius <= 0:
        return mv
    mag = mv.pow(2).sum(-1)
    best_mag, best_mv = mag.clone(), mv.clone()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            sm = torch.roll(mag, shifts=(dy, dx), dims=(-2, -1))
            sv = torch.roll(mv, shifts=(dy, dx), dims=(-3, -2))
            take = sm > best_mag
            best_mag = torch.where(take, sm, best_mag)
            best_mv = torch.where(take.unsqueeze(-1), sv, best_mv)
    return best_mv


def adaptive_warp_steps(A, B, mvA, mvB, base_steps, *, max_factor=2,
                        target_motion_px=1.0, target_log_step=0.15):
    """Choose temporal knots for one keyframe gap from its observed difficulty.

    The DVS processor solves every threshold-crossing time inside each interval,
    so this function only needs to keep the piecewise log-intensity trajectory
    sufficiently fine. It increases ``base_steps`` when the 99th-percentile
    endpoint motion exceeds roughly one pixel per interval, or when the robust
    (90th-percentile) endpoint log-luminance change exceeds one contrast
    threshold per interval. The result is globally capped for predictable GPU
    memory use.

    Percentiles deliberately ignore a handful of renderer outliers while still
    responding to a moving manipulation target or a broad HDR reflection.
    """
    base_steps = int(base_steps)
    if base_steps <= 1 or int(max_factor) <= 1:
        return max(1, base_steps)
    max_steps = max(base_steps, base_steps * int(max_factor))

    motion = torch.maximum(
        torch.linalg.vector_norm(torch.nan_to_num(mvA), dim=-1),
        torch.linalg.vector_norm(torch.nan_to_num(mvB), dim=-1),
    ).float().reshape(-1)
    motion_q = torch.quantile(motion, 0.99) if motion.numel() else motion.new_tensor(0.0)
    motion_steps = int(torch.ceil(motion_q / max(float(target_motion_px), 1e-6)).item())

    def log_luminance(frame):
        rgb = torch.nan_to_num(frame[..., :3].float()).clamp_min(0.0)
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return torch.log(lum + 1e-5)

    log_change = (log_luminance(B) - log_luminance(A)).abs().reshape(-1)
    log_q = torch.quantile(log_change, 0.90) if log_change.numel() else log_change.new_tensor(0.0)
    photometric_steps = int(
        torch.ceil(log_q / max(float(target_log_step), 1e-6)).item()
    )
    return min(max_steps, max(base_steps, motion_steps, photometric_steps))


def bidir_warp_gap(A, B, mvA, mvB, K, composite="b_primary",
                   depthA=None, depthB=None, hw=1.0, beta=12.0,
                   fill_holes=True, covis_z=False, hole_fill=None, mv_dilate=0,
                   return_validity=False, return_confidence=False,
                   depth_abs_tol=0.05, depth_rel_tol=0.05,
                   flow_abs_tol=1.0, flow_rel_tol=0.25, valid_margin=1):
    """Single-mv-per-gap bidirectional warp (the 125/250Hz-mv case).

    One real mv field per keyframe gap (the WHOLE-gap displacement, convention
    earlier-pos = pos + mv), so motion is taken as straight-line across the gap:
    forward-warp the previous keyframe A by ``f`` using ``mvA``, backward-warp the
    next keyframe B by ``1-f`` using ``mvB``, then composite. This is the warp
    used by both the offline naive-125 comparison and the e2e benchmark.

    Composite is B-primary: the backward-warped B is the source of truth
    wherever B has content; A is used ONLY to fill B's disocclusion holes (see
    ``covis_z`` for the depth-aware variant). Pixels occluded in BOTH default to
    black (genuinely unknown — no colour is fabricated) unless ``hole_fill``
    supplies a constant.

    Depth, when given, resolves COLLISIONS inside each splat: the importance
    becomes ``1/depth`` so the nearer (foreground) source wins when several source
    pixels land on the same target. Without depth the importance falls back to
    displacement magnitude (fast-moving as a foreground proxy).

    ``covis_z`` (needs depth): if True, the co-visible region is no longer "always
    B" — instead each keyframe's depth is warped alongside its colour and the
    pixel keeps whichever of A/B has the NEARER warped surface. If False the
    co-visible region stays B-primary.

    ``composite="log_blend"`` is the event-camera path. In regions where both
    keyframes agree in depth and motion, it interpolates log luminance linearly
    in time. This prevents a future keyframe's HDR lighting from appearing in
    the first synthetic frame as a step change. Occlusions, holes, and pixels
    whose warped depth/flow disagree are marked invalid. With
    ``return_validity=True`` the function returns ``(frames, valid_masks)``;
    consumers should suppress signal events in invalid regions. Setting
    ``return_confidence=True`` additionally returns a soft per-pixel warp
    confidence in ``[0,1]``. It is intentionally metadata, not a hard gate:
    moving objects with imperfect endpoint agreement remain in the event stream.

    A,B  : (H,W,C) OR batched (N,H,W,C) float on GPU.  mvA,mvB : (..,H,W,2).
    depthA,depthB : (..,H,W) per-pixel depth of keyframe A / B (metres).
    ``mv_dilate`` (px): grow each mv field over its anti-aliased edge fringe
    before splatting (see :func:`dilate_mv`); removes the boundary ghost outline.
    Every env, both splat directions and all K-1 intermediate frames run as ONE
    batch (M = N*(K-1) per direction). Returns the K-1 intermediates at fractions
    ``1/K..(K-1)/K``, each (H,W,C) for single input or (N,H,W,C) for batched."""
    if composite not in ("b_primary", "log_blend"):
        raise ValueError(
            f"unknown composite {composite!r}; expected 'b_primary' or 'log_blend'"
        )
    single = A.dim() == 3
    if single:
        A, B, mvA, mvB = A[None], B[None], mvA[None], mvB[None]
        if depthA is not None:
            depthA, depthB = depthA[None], depthB[None]
    if mv_dilate:
        mvA, mvB = dilate_mv(mvA, mv_dilate), dilate_mv(mvB, mv_dilate)
    if K == 1:
        if return_confidence:
            return [], [], []
        return ([], []) if return_validity else []
    N, H, W, C = A.shape
    dev = A.device
    Kn = K - 1
    Ai = A.permute(0, 3, 1, 2)                     # (N,C,H,W)
    Bi = B.permute(0, 3, 1, 2)
    use_z = depthA is not None and depthB is not None
    z_covis = use_z and covis_z
    if use_z:
        zA = depthA if depthA.dim() == 3 else depthA[..., 0]   # (N,H,W)
        zB = depthB if depthB.dim() == 3 else depthB[..., 0]
        impA0 = 1.0 / (zA + 1e-3)                  # near -> large importance -> wins collision
        impB0 = 1.0 / (zB + 1e-3)
        # Carry depth through the same splat as colour. Besides resolving
        # collisions, it lets log_blend reject cross-surface correspondences.
        Ai = torch.cat([Ai, zA.unsqueeze(1)], dim=1)
        Bi = torch.cat([Bi, zB.unsqueeze(1)], dim=1)
    # Warped motion must agree at a common intermediate pixel. This detects
    # mismatched foreground/background splats even when their depths are close.
    Ai = torch.cat([Ai, mvA.permute(0, 3, 1, 2)], dim=1)
    Bi = torch.cat([Bi, mvB.permute(0, 3, 1, 2)], dim=1)
    Cc = Ai.shape[1]
    fs = torch.arange(1, K, device=dev, dtype=A.dtype) / K      # (Kn,) fractions
    dA = (-fs).view(1, Kn, 1, 1, 1) * mvA.unsqueeze(1)         # (N,Kn,H,W,2)
    dB = (1.0 - fs).view(1, Kn, 1, 1, 1) * mvB.unsqueeze(1)
    M = N * Kn
    Arep = Ai.unsqueeze(1).expand(N, Kn, Cc, H, W).reshape(M, Cc, H, W)
    Brep = Bi.unsqueeze(1).expand(N, Kn, Cc, H, W).reshape(M, Cc, H, W)
    dAx, dAy = dA[..., 0].reshape(M, H, W), dA[..., 1].reshape(M, H, W)
    dBx, dBy = dB[..., 0].reshape(M, H, W), dB[..., 1].reshape(M, H, W)
    if use_z:
        impA = impA0.unsqueeze(1).expand(N, Kn, H, W).reshape(M, H, W)
        impB = impB0.unsqueeze(1).expand(N, Kn, H, W).reshape(M, H, W)
    else:
        impA = torch.sqrt(dAx ** 2 + dAy ** 2)
        impB = torch.sqrt(dBx ** 2 + dBy ** 2)
    wA_, hA = _splat_batch(Arep, dAx, dAy, impA, hw=hw, beta=beta)   # (M,Cc,H,W),(M,H,W)
    wB_, hB = _splat_batch(Brep, dBx, dBy, impB, hw=hw, beta=beta)
    wA, wB = wA_[:, :C], wB_[:, :C]
    aux = C
    if use_z:
        wzA, wzB = wA_[:, aux], wB_[:, aux]
        aux += 1
    else:
        wzA = wzB = None
    wmvA, wmvB = wA_[:, aux:aux + 2], wB_[:, aux:aux + 2]

    has_a, has_b = ~hA, ~hB
    both = has_a & has_b
    only_a = (has_a & ~has_b).unsqueeze(1)
    neither = (~has_a & ~has_b).unsqueeze(1)

    flow_error = torch.linalg.vector_norm(wmvA - wmvB, dim=1)
    flow_scale = torch.maximum(
        torch.linalg.vector_norm(wmvA, dim=1),
        torch.linalg.vector_norm(wmvB, dim=1),
    )
    flow_ok = flow_error <= (flow_abs_tol + flow_rel_tol * flow_scale)
    if use_z:
        depth_scale = torch.minimum(wzA.abs(), wzB.abs())
        depth_error = (wzA - wzB).abs()
        depth_tol = depth_abs_tol + depth_rel_tol * depth_scale
        depth_ok = depth_error <= depth_tol
    else:
        depth_error = None
        depth_tol = None
        depth_ok = torch.ones_like(flow_ok)
    # Agreement is useful for deciding whether A/B describe the same surface,
    # but it must not be a hard event-validity gate. Accelerating/rotating task
    # objects naturally have different endpoint flow and were otherwise erased
    # from the event stream. When depth exists it is the stronger surface cue;
    # flow is the fallback only for RGB-only callers.
    same_surface = both & (depth_ok if use_z else flow_ok)

    # Only a true bidirectional coverage hole is unobservable. One-sided
    # disocclusions and inconsistent dynamic surfaces still carry real image
    # content and must remain capable of producing events.
    valid = has_a | has_b

    # Soft quality is separate from hard observability. Endpoint disagreement
    # lowers confidence continuously, while a one-sided disocclusion still gets
    # a usable (but explicitly uncertain) score. Depth is the stronger cue when
    # available; flow contributes more gently because acceleration and rotation
    # naturally make endpoint motion vectors disagree on real task objects.
    flow_tol = flow_abs_tol + flow_rel_tol * flow_scale
    flow_conf = torch.exp(-flow_error / flow_tol.clamp_min(1e-6))
    if use_z:
        depth_conf = torch.exp(-depth_error / depth_tol.clamp_min(1e-6))
        agreement_conf = torch.sqrt(depth_conf * torch.sqrt(flow_conf))
    else:
        agreement_conf = flow_conf
    # A visible pixel always has at least one actual rendered endpoint, so its
    # score must not approach zero merely because a rotating/accelerating object
    # disagrees across endpoints. Reserve [0,0.5) for unobservable regions and
    # map endpoint agreement into [0.5,1]. This keeps confidence weighting from
    # erasing exactly the cup/robot boundaries needed by a manipulation policy.
    confidence = torch.where(
        both,
        0.5 + 0.5 * agreement_conf,
        torch.full_like(agreement_conf, 0.5),
    )
    confidence = torch.where(has_a | has_b, confidence, torch.zeros_like(confidence))

    if valid_margin > 0:
        # A double-hole fill can splat a bright/dark fringe into its neighbours.
        # Erode only around those genuinely unobserved pixels, not around every
        # depth/flow disagreement on a moving object.
        invalid = F.max_pool2d(
            neither.float(),
            kernel_size=2 * valid_margin + 1,
            stride=1,
            padding=valid_margin,
        ).squeeze(1) > 0
        valid = ~invalid
        confidence = torch.where(valid, confidence, torch.zeros_like(confidence))

    alpha = fs.repeat(N).view(M, 1, 1, 1)
    if composite == "log_blend":
        # Blend chromaticity linearly, but force luminance to follow a geometric
        # (log-linear) path. The event model therefore sees a constant contrast
        # rate rather than an HDR step at the first intermediate frame.
        eps = 1e-5
        wA_pos, wB_pos = wA.clamp_min(0.0), wB.clamp_min(0.0)
        lumA = 0.2126 * wA_pos[:, 0] + 0.7152 * wA_pos[:, 1] + 0.0722 * wA_pos[:, 2]
        lumB = 0.2126 * wB_pos[:, 0] + 0.7152 * wB_pos[:, 1] + 0.0722 * wB_pos[:, 2]
        target_lum = torch.exp(
            (1.0 - alpha[:, 0]) * torch.log(lumA + eps)
            + alpha[:, 0] * torch.log(lumB + eps)
        ) - eps
        blended = (1.0 - alpha) * wA_pos + alpha * wB_pos
        blended_lum = (
            0.2126 * blended[:, 0] + 0.7152 * blended[:, 1] + 0.0722 * blended[:, 2]
        )
        blended = blended * (target_lum / (blended_lum + eps)).unsqueeze(1)
        m = torch.where(same_surface.unsqueeze(1), blended, wB)
    else:
        m = wB.clone()                            # legacy B-primary behaviour

    if z_covis:
        nearer_a = both & ~same_surface & (wzA < wzB)
        m = torch.where(nearer_a.unsqueeze(1), wA, m)
    if fill_holes:
        m = torch.where(only_a, wA, m)            # B's disocclusion hole -> fill from A
    # double-occlusion: black by default (unknown, no fabrication); or a constant
    # ``hole_fill`` (e.g. the background level) so holes vanish on a uniform backdrop.
    fill = torch.zeros_like(m) if hole_fill is None else torch.full_like(m, float(hole_fill))
    m = torch.where(neither, fill, m)
    m = m.reshape(N, Kn, C, H, W).permute(0, 1, 3, 4, 2)        # (N,Kn,H,W,C)
    valid = valid.reshape(N, Kn, H, W)
    confidence = confidence.reshape(N, Kn, H, W).clamp_(0.0, 1.0)
    outs = [m[:, i] for i in range(Kn)]
    valid_outs = [valid[:, i] for i in range(Kn)]
    confidence_outs = [confidence[:, i] for i in range(Kn)]
    if single:
        outs = [o[0] for o in outs]
        valid_outs = [v[0] for v in valid_outs]
        confidence_outs = [q[0] for q in confidence_outs]
    if return_confidence:
        return outs, valid_outs, confidence_outs
    return (outs, valid_outs) if return_validity else outs


def _sample_field(field, pos):
    """Bilinearly sample a (H,W,2) field at sub-pixel positions ``pos`` (H,W,2)."""
    H, W, _ = field.shape
    gx = pos[..., 0] / (W - 1) * 2 - 1
    gy = pos[..., 1] / (H - 1) * 2 - 1
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    f = field.permute(2, 0, 1).unsqueeze(0)
    s = F.grid_sample(f, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return s[0].permute(1, 2, 0)


@register("dense_bidir")
class DenseBidirWarp(FrameInterpolator):
    """Bidirectional warp driven by the FULL 1000 Hz mv sequence in a keyframe gap.

    Each pixel is traced along its true (curved) trajectory by integrating the
    per-1ms mv fields one small step at a time, RESAMPLING the mv at the moving
    sub-pixel position each step (Lagrangian). The previous keyframe A is traced
    FORWARD and the next keyframe B BACKWARD to the same intermediate time, then
    composited so each fills the other's disocclusion holes. Regions occluded in
    BOTH are left black (genuinely unknown). No background fabrication.

    Aux: ``rgb_next`` = B (next keyframe), ``mv_seq`` = (K,H,W,2) the per-step mv
    fields for frames k+1..k+K (mv convention: earlier-position = pos + mv)."""

    def __init__(self, frames_per_keyframe: int = 8, splat_hw: float = 1.0, beta: float = 12.0,
                 composite: str = "b_primary", dense: bool = True):
        super().__init__(frames_per_keyframe)
        self.splat_hw = float(splat_hw)
        self.beta = float(beta)
        # how to merge the forward-A and backward-B warps in co-visible regions:
        #   "avg"       - average 0.5*(wA+wB)
        #   "b_primary" - keep the (accurate) backward-B; use A only to fill B's holes
        self.composite = str(composite)
        # dense=True : trace the true curved path through the per-1ms mv fields (needs 1000Hz mv)
        # dense=False: collapse the gap to ONE averaged mv field -> straight-line / constant-velocity
        #              motion. This is what you'd get with mv rendered only at 125Hz (one field/gap).
        self.dense = bool(dense)

    def __call__(self, rgb: torch.Tensor, *, rgb_next=None, mv_seq=None, **aux) -> list[torch.Tensor]:
        if rgb_next is None or mv_seq is None:
            raise ValueError("dense_bidir needs rgb_next and mv_seq (K per-step mv fields)")
        if self.num_intermediate == 0:
            return []
        A = rgb[0].permute(2, 0, 1).float()
        B = rgb_next[0].permute(2, 0, 1).float()
        mv = mv_seq.float()
        if mv.dim() == 5:                       # (1,K,H,W,2) -> (K,H,W,2)
            mv = mv[0]
        if not self.dense:                      # 125Hz mv: one averaged field -> straight-line motion
            mv = mv.mean(dim=0, keepdim=True).expand_as(mv).contiguous()
        K = self.K
        C, H, W = A.shape
        dev = A.device
        ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        idx = torch.stack((xs.float(), ys.float()), dim=-1)          # identity grid (H,W,2)

        # Trace A FORWARD, snapshot position after i steps (i = 1..K-1)
        posA = idx.clone(); snapA = {0: idx}
        for j in range(0, K - 1):
            posA = posA - _sample_field(mv[j], posA)                 # forward step ~ -backward mv
            snapA[j + 1] = posA.clone()
        # Trace B BACKWARD, snapshot position at frame k+i (i = K-1..1)
        posB = idx.clone(); snapB = {K: idx}
        for j in range(K - 1, 0, -1):
            posB = posB + _sample_field(mv[j], posB)                 # backward step
            snapB[j] = posB.clone()

        out: list[torch.Tensor] = []
        for i in range(1, K):
            pA, pB = snapA[i], snapB[i]
            dAx, dAy = pA[..., 0] - idx[..., 0], pA[..., 1] - idx[..., 1]
            dBx, dBy = pB[..., 0] - idx[..., 0], pB[..., 1] - idx[..., 1]
            impA = torch.sqrt(dAx ** 2 + dAy ** 2)
            impB = torch.sqrt(dBx ** 2 + dBy ** 2)
            wA, hA = _splat(A, dAx, dAy, imp=impA, hw=self.splat_hw, beta=self.beta)
            wB, hB = _splat(B, dBx, dBy, imp=impB, hw=self.splat_hw, beta=self.beta)
            only_a = (~hA) & hB
            neither = hA & hB
            m = wB.clone()
            if self.composite == "avg":
                both = (~hA) & (~hB)
                m[:, both] = 0.5 * (wA + wB)[:, both]                # co-visible -> average
            m[:, only_a] = wA[:, only_a]                            # B's disocclusion hole -> fill from A
            m[:, neither] = 0.0                                      # double-occlusion -> black
            out.append(m.permute(1, 2, 0).unsqueeze(0).to(rgb.dtype))
        return out
