"""
processor.py
============
``BatchedMultiCamProcessor`` — converts a batched RGB tensor (all envs of ONE
camera) into DVS events using the standard log-intensity-difference model, and
forwards them to a :class:`~dvs_gen.dvs.recorder.GeneralDVSRecorder`.

Event model: per pixel, every ``±threshold`` crossing in ``log(intensity)`` emits
an event. Crossing times are linearly interpolated inside the frame interval and
the reference advances by exact threshold increments, preserving the residual.
Pure torch — no Omniverse dependency.
"""
import torch
import torch.nn.functional as F

from .recorder import GeneralDVSRecorder


class BatchedMultiCamProcessor:
    """Processes batched RGB tensors for ONE specific camera across ALL envs."""
    def __init__(self, recorder: GeneralDVSRecorder, camera_name: str, threshold: float = 0.15,
                 noise=None, hybrid_gate_gain: float = 0.0,
                 hybrid_support_radius: int = 2):
        self.recorder = recorder
        self.camera_name = camera_name
        self.threshold = threshold
        # optional DVSNoiseModel (dvs_gen.dvs.noise). None = the ideal clean model.
        self.noise = noise
        # v4 opt-in: a low-confidence synthetic warp sample must exceed a
        # slightly stronger contrast gate before its already-quantized sensor
        # crossings are recorded. Rejected weak crossings are consumed from the
        # reference, so they cannot reappear as a keyframe burst. Strong events
        # and same-polarity neighbours remain intact to preserve object edges.
        self.hybrid_gate_gain = max(0.0, float(hybrid_gate_gain))
        self.hybrid_support_radius = max(0, int(hybrid_support_radius))
        self.ref_log_intensity = None
        self.prev_log_intensity = None
        self.prev_time = None
        self.prev_valid_mask = None
        self.prev_confidence = None
        self.needs_reset_mask = None # Tracks which envs need a new reference frame
        # opt-in: keep this call's (pos_mask, neg_mask) so a caller can render the
        # per-frame event image aligned to the warped RGB frame. Off = zero overhead.
        self.stash_events = False
        self.last_masks = None

    def reset_envs(self, env_ids: torch.Tensor):
        """Flags specific environments to grab a fresh reference frame."""
        if self.needs_reset_mask is not None and len(env_ids) > 0:
            self.needs_reset_mask[env_ids] = True
            if self.noise is not None:
                self.noise.reset_envs(env_ids)         # re-seed the bandwidth filter state

    def __call__(self, rgb_batch: torch.Tensor, current_time: float,
                 valid_mask: torch.Tensor | None = None,
                 confidence: torch.Tensor | None = None):
        # rgb_batch: (num_envs, H, W, C)
        num_envs = rgb_batch.shape[0]
        device = rgb_batch.device
        if self.stash_events:
            self.last_masks = None      # cleared per call; set below once masks exist

        if rgb_batch.shape[-1] == 4:
            rgb_batch = rgb_batch[..., :3]

        intensity = (0.2126 * rgb_batch[..., 0] + 0.7152 * rgb_batch[..., 1] + 0.0722 * rgb_batch[..., 2])
        log_intensity = torch.log(intensity + 1e-5)
        if valid_mask is None:
            valid_mask = torch.ones_like(log_intensity, dtype=torch.bool)
        else:
            if valid_mask.dim() == log_intensity.dim() + 1 and valid_mask.shape[-1] == 1:
                valid_mask = valid_mask[..., 0]
            if valid_mask.shape != log_intensity.shape:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} does not match "
                    f"intensity shape {tuple(log_intensity.shape)}"
                )
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        if confidence is None:
            confidence = torch.ones_like(log_intensity)
        else:
            if confidence.dim() == log_intensity.dim() + 1 and confidence.shape[-1] == 1:
                confidence = confidence[..., 0]
            if confidence.shape != log_intensity.shape:
                raise ValueError(
                    f"confidence shape {tuple(confidence.shape)} does not match "
                    f"intensity shape {tuple(log_intensity.shape)}"
                )
            confidence = confidence.to(device=device, dtype=log_intensity.dtype)
            confidence = confidence.clamp(0.0, 1.0)
        confidence = torch.where(valid_mask, confidence, torch.zeros_like(confidence))

        # Invalid warp pixels have no trustworthy irradiance measurement. Keep
        # their last valid photoreceptor state rather than turning a splat hole
        # or a cross-surface mismatch into signal events.
        if self.prev_log_intensity is not None:
            log_intensity = torch.where(valid_mask, log_intensity, self.prev_log_intensity)

        # Intensity-dependent photoreceptor bandwidth (noise model hook 0): lowpass
        # the log frame BEFORE the reference logic so the whole event model sees the
        # filtered signal. cutoff_hz == 0 (default) returns it untouched.
        if self.noise is not None:
            log_intensity = self.noise.bandwidth_filter(log_intensity, intensity, current_time)
            if self.prev_log_intensity is not None:
                log_intensity = torch.where(valid_mask, log_intensity, self.prev_log_intensity)

        # Initialization
        if self.ref_log_intensity is None:
            self.ref_log_intensity = log_intensity.clone()
            self.prev_log_intensity = log_intensity.clone()
            self.prev_time = float(current_time)
            self.prev_valid_mask = valid_mask.clone()
            self.prev_confidence = confidence.clone()
            self.needs_reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
            return

        # 1. Update references for environments that just reset
        if self.needs_reset_mask.any():
            self.ref_log_intensity[self.needs_reset_mask] = log_intensity[self.needs_reset_mask]
            self.prev_log_intensity[self.needs_reset_mask] = log_intensity[self.needs_reset_mask]
            self.prev_confidence[self.needs_reset_mask] = confidence[self.needs_reset_mask]
            reset_mask = self.needs_reset_mask.clone()
            self.needs_reset_mask.fill_(False)
            # We don't generate events for the reset frame itself
        else:
            reset_mask = None

        # A pixel that becomes trustworthy again after one or more invalid warp
        # samples is re-anchored silently. Emitting its accumulated difference at
        # the recovery frame would create exactly the periodic event burst the
        # validity mask is intended to prevent.
        revalidated = valid_mask & ~self.prev_valid_mask
        if revalidated.any():
            self.ref_log_intensity[revalidated] = log_intensity[revalidated]
            self.prev_log_intensity[revalidated] = log_intensity[revalidated]
            self.prev_confidence[revalidated] = confidence[revalidated]

        # An event spans the interval between two irradiance samples, so its
        # quality cannot exceed the weaker endpoint. This score is metadata only:
        # unlike ``valid_mask`` it never suppresses a real event.
        event_confidence = torch.minimum(self.prev_confidence, confidence)

        # 2. Compute differences. With a noise model, use PER-PIXEL thresholds
        # (fixed-pattern mismatch); otherwise the single scalar threshold.
        diff = log_intensity - self.ref_log_intensity
        if self.noise is not None:
            th_on, th_off = self.noise.thresholds(log_intensity.shape[1:], device)
            th_on_full = th_on.unsqueeze(0).expand_as(diff)
            th_off_full = th_off.unsqueeze(0).expand_as(diff)
        else:
            th_on_full = torch.full_like(diff, self.threshold)
            th_off_full = th_on_full

        # One sampled frame may cross several contrast thresholds. Keep all of
        # them, rather than collapsing the entire change into one boolean event.
        n_pos = torch.floor(torch.clamp_min(diff, 0.0) / th_on_full).to(torch.int64)
        n_neg = torch.floor(torch.clamp_min(-diff, 0.0) / th_off_full).to(torch.int64)
        n_pos[~valid_mask] = 0
        n_neg[~valid_mask] = 0
        if reset_mask is not None:
            n_pos[reset_mask] = 0
            n_neg[reset_mask] = 0

        # Hybrid v4 keeps the physical threshold grid and reference update from
        # v2/v3, but declines weak crossings that exist only in an uncertain
        # synthetic sample. Using the CURRENT sample confidence means a real
        # keyframe (q=1) is never gated because the preceding warp was uncertain.
        emit_pos, emit_neg = n_pos, n_neg
        if self.hybrid_gate_gain > 0.0:
            gate_scale = 1.0 + self.hybrid_gate_gain * (1.0 - confidence)
            strong_pos = diff >= th_on_full * gate_scale
            strong_neg = -diff >= th_off_full * gate_scale
            if self.hybrid_support_radius > 0:
                radius = self.hybrid_support_radius
                kernel = 2 * radius + 1
                strong_pos = F.max_pool2d(
                    strong_pos.float().unsqueeze(1), kernel, stride=1, padding=radius
                ).squeeze(1) > 0
                strong_neg = F.max_pool2d(
                    strong_neg.float().unsqueeze(1), kernel, stride=1, padding=radius
                ).squeeze(1) > 0
            emit_pos = torch.where(strong_pos, n_pos, torch.zeros_like(n_pos))
            emit_neg = torch.where(strong_neg, n_neg, torch.zeros_like(n_neg))

        # Advance by exact threshold increments. The sub-threshold remainder is
        # retained in (log_intensity - reference) for the next frame.
        ref_before = self.ref_log_intensity.clone()
        self.ref_log_intensity = (
            self.ref_log_intensity
            + n_pos.to(log_intensity.dtype) * th_on_full
            - n_neg.to(log_intensity.dtype) * th_off_full
        )

        # Emit signal crossings with interpolated timestamps. The noisy path runs
        # crossing layers in order so refractory filtering remains physically
        # chronological for each pixel.
        if self.noise is not None:
            pos_seen, neg_seen = self._record_crossing_layers(
                emit_pos, emit_neg, th_on_full, th_off_full, ref_before,
                log_intensity, current_time, event_confidence,
            )
            empty = torch.zeros_like(n_pos, dtype=torch.bool)
            noise_pos, noise_neg = self.noise.apply(empty, empty, intensity, current_time)
            noise_pos &= valid_mask
            noise_neg &= valid_mask
            self._record_masks(noise_pos, noise_neg, current_time, event_confidence)
            pos_seen |= noise_pos
            neg_seen |= noise_neg
        else:
            pos_seen, neg_seen = self._record_expanded_crossings(
                emit_pos, emit_neg, th_on_full, th_off_full, ref_before,
                log_intensity, current_time, event_confidence,
            )

        if self.stash_events:               # final recorded masks (incl. noise) for the event image
            self.last_masks = (pos_seen, neg_seen)

        self.prev_log_intensity = torch.where(
            valid_mask, log_intensity, self.prev_log_intensity
        )
        self.prev_valid_mask = valid_mask.clone()
        self.prev_confidence = confidence.clone()
        self.prev_time = float(current_time)

    def _crossing_time(self, target, current, previous, current_time):
        """Linearly interpolate the time at which ``target`` was crossed."""
        dt = max(0.0, float(current_time) - float(self.prev_time))
        # Time is stored as float64 in HDF5. Compute it in float64 here as well;
        # float32 simulation timestamps can otherwise differ by sub-microseconds
        # at adjacent frame boundaries and make an otherwise ordered stream look
        # non-monotonic.
        target = target.to(torch.float64)
        current = current.to(torch.float64)
        previous = previous.to(torch.float64)
        denom = current - previous
        alpha = torch.where(
            denom.abs() > 1e-12,
            (target - previous) / denom,
            torch.ones_like(target),
        ).clamp(0.0, 1.0)
        return torch.as_tensor(
            float(self.prev_time), dtype=torch.float64, device=target.device
        ) + alpha * dt

    def _expanded_polarity(self, counts, thresholds, ref_before, current,
                           current_time, polarity):
        active = counts > 0
        if not active.any():
            empty_i = torch.empty(0, dtype=torch.int64, device=counts.device)
            empty_t = torch.empty(0, dtype=current.dtype, device=current.device)
            return empty_i, empty_i, empty_i, empty_t

        envs, ys, xs = torch.where(active)
        per_pixel = counts[active]
        owners = torch.repeat_interleave(
            torch.arange(per_pixel.numel(), device=counts.device), per_pixel
        )
        starts = torch.cumsum(per_pixel, 0) - per_pixel
        ordinal = (
            torch.arange(owners.numel(), device=counts.device)
            - torch.repeat_interleave(starts, per_pixel)
            + 1
        )
        envs = envs[owners]
        ys = ys[owners]
        xs = xs[owners]
        target = (
            ref_before[active][owners]
            + polarity * ordinal.to(current.dtype) * thresholds[active][owners]
        )
        times = self._crossing_time(
            target,
            current[envs, ys, xs],
            self.prev_log_intensity[envs, ys, xs],
            current_time,
        )
        return envs, ys, xs, times

    def _record_expanded_crossings(self, n_pos, n_neg, th_on, th_off,
                                   ref_before, current, current_time, confidence):
        ep, yp, xp, tp = self._expanded_polarity(
            n_pos, th_on, ref_before, current, current_time, 1
        )
        en, yn, xn, tn = self._expanded_polarity(
            n_neg, th_off, ref_before, current, current_time, -1
        )
        envs = torch.cat([ep, en])
        ys = torch.cat([yp, yn])
        xs = torch.cat([xp, xn])
        times = torch.cat([tp, tn])
        ps = torch.cat([
            torch.ones(ep.numel(), dtype=torch.int8, device=current.device),
            -torch.ones(en.numel(), dtype=torch.int8, device=current.device),
        ])
        if times.numel() > 0:
            order = torch.argsort(times)
            quality = confidence[envs, ys, xs]
            self.recorder.record(
                self.camera_name, envs[order], xs[order], ys[order], ps[order],
                times[order], confidence=quality[order],
            )
        return n_pos > 0, n_neg > 0

    def _record_crossing_layers(self, n_pos, n_neg, th_on, th_off,
                                ref_before, current, current_time, confidence):
        pos_seen = torch.zeros_like(n_pos, dtype=torch.bool)
        neg_seen = torch.zeros_like(n_neg, dtype=torch.bool)
        batches = []
        max_crossings = int(torch.maximum(n_pos.max(), n_neg.max()).item())
        for crossing in range(1, max_crossings + 1):
            pos_mask = n_pos >= crossing
            neg_mask = n_neg >= crossing
            pos_target = ref_before + crossing * th_on
            neg_target = ref_before - crossing * th_off
            pos_time = self._crossing_time(
                pos_target, current, self.prev_log_intensity, current_time
            )
            neg_time = self._crossing_time(
                neg_target, current, self.prev_log_intensity, current_time
            )
            event_time = torch.where(pos_mask, pos_time, neg_time)
            pos_mask, neg_mask = self.noise.apply_signal_refractory(
                pos_mask, neg_mask, event_time
            )
            batch = self._events_from_masks(pos_mask, neg_mask, event_time)
            if batch is not None:
                batches.append(batch)
            pos_seen |= pos_mask
            neg_seen |= neg_mask
        if batches:
            envs, xs, ys, ps, times = (
                torch.cat([batch[i] for batch in batches]) for i in range(5)
            )
            order = torch.argsort(times)
            quality = confidence[envs, ys, xs]
            self.recorder.record(
                self.camera_name, envs[order], xs[order], ys[order], ps[order],
                times[order], confidence=quality[order],
            )
        return pos_seen, neg_seen

    def _events_from_masks(self, pos_mask, neg_mask, event_time):
        if not (pos_mask.any() or neg_mask.any()):
            return None
        envs_pos, ys_pos, xs_pos = torch.where(pos_mask)
        envs_neg, ys_neg, xs_neg = torch.where(neg_mask)
        all_envs = torch.cat([envs_pos, envs_neg])
        all_xs = torch.cat([xs_pos, xs_neg])
        all_ys = torch.cat([ys_pos, ys_neg])
        all_ps = torch.cat([
            torch.ones(xs_pos.shape[0], dtype=torch.int8, device=pos_mask.device),
            -torch.ones(xs_neg.shape[0], dtype=torch.int8, device=pos_mask.device),
        ])
        if torch.is_tensor(event_time):
            all_t = torch.cat([
                event_time[envs_pos, ys_pos, xs_pos],
                event_time[envs_neg, ys_neg, xs_neg],
            ])
        else:
            all_t = torch.full(
                (all_xs.numel(),), float(event_time), dtype=torch.float64,
                device=pos_mask.device,
            )
        return all_envs, all_xs, all_ys, all_ps, all_t

    def _record_masks(self, pos_mask, neg_mask, event_time, confidence):
        batch = self._events_from_masks(pos_mask, neg_mask, event_time)
        if batch is not None:
            all_envs, all_xs, all_ys, all_ps, all_t = batch
            order = torch.argsort(all_t)
            quality = confidence[all_envs, all_ys, all_xs]
            self.recorder.record(
                self.camera_name, all_envs[order], all_xs[order], all_ys[order],
                all_ps[order], all_t[order], confidence=quality[order],
            )
