"""
recorder.py
===========
``GeneralDVSRecorder`` — a thread-safe, multi-env / multi-camera DVS event
buffer that flushes one HDF5 file per (env, episode).

This is pure Python/torch/numpy/h5py — it has no Omniverse dependency, so it can
be imported and unit-tested outside Isaac Sim.

HDF5 layout (one file ``env{env_id}_ep{episode_idx}.h5`` per environment)::

    @event_time_origin_s  float64  absolute Isaac time corresponding to episode t=0
    @evis_mode            string   v2_balanced / v3_adaptive / v4_hybrid
    /<camera_name>/x   uint16   (gzip)  pixel x
    /<camera_name>/y   uint16   (gzip)  pixel y
    /<camera_name>/t   float64  (gzip)  timestamp (seconds)
    /<camera_name>/p   int8     (gzip)  polarity  (+1 = ON, -1 = OFF)
    /<camera_name>/q   float16  (gzip)  warp confidence (0=hole, visible>=0.5)
"""
import os
import threading
from collections import defaultdict

import numpy as np
import torch
import h5py


class GeneralDVSRecorder:
    """Thread-safe recorder for multiple envs, multiple cameras, varying episodes."""
    def __init__(self, output_dir: str = "/tmp/dvs_dataset", compression="gzip"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._lock = threading.Lock()
        # HDF5 filter for the event datasets: "gzip" (small files), "lzf" (fast),
        # or None (uncompressed — maximally robust for very large streams, no
        # filter that can fail on read). Datasets are written in blocks so a
        # multi-million-event episode never does one monolithic compressed write.
        self.compression = compression

        # Nested dictionary: env_id -> camera_name -> list of events
        self._events = defaultdict(lambda: defaultdict(list))

    def record(self, camera_name: str, env_ids: torch.Tensor, xs: torch.Tensor,
               ys: torch.Tensor, ps: torch.Tensor, t, confidence=None):
        """Buffer an event batch.

        ``t`` may be one scalar timestamp (legacy callers) or one timestamp per
        event.  Per-event timestamps are required when a single intensity update
        crosses several contrast thresholds.
        """
        if xs.numel() == 0: return

        # One GPU->CPU transfer per array, then store whole array-chunks per env
        # (NO per-event Python loop). Chunks are concatenated lazily at flush.
        e_np = env_ids.cpu().numpy()
        x_np = xs.cpu().numpy().astype(np.uint16)
        y_np = ys.cpu().numpy().astype(np.uint16)
        p_np = ps.cpu().numpy().astype(np.int8)
        if torch.is_tensor(t):
            t_np = t.detach().cpu().numpy().astype(np.float64)
        else:
            t_np = np.asarray(t, dtype=np.float64)
        if t_np.ndim == 0:
            t_np = np.full(x_np.shape, float(t_np), np.float64)
        else:
            t_np = t_np.reshape(-1)
            if t_np.shape != x_np.shape:
                raise ValueError(
                    f"Expected one timestamp per event ({x_np.shape}), got {t_np.shape}."
                )
        if confidence is None:
            q_np = np.ones(x_np.shape, dtype=np.float16)
        elif torch.is_tensor(confidence):
            q_np = confidence.detach().cpu().numpy().astype(np.float16).reshape(-1)
        else:
            q_np = np.asarray(confidence, dtype=np.float16)
            if q_np.ndim == 0:
                q_np = np.full(x_np.shape, float(q_np), dtype=np.float16)
            else:
                q_np = q_np.reshape(-1)
        if q_np.shape != x_np.shape:
            raise ValueError(
                f"Expected one confidence per event ({x_np.shape}), got {q_np.shape}."
            )
        q_np = np.clip(q_np, 0.0, 1.0).astype(np.float16, copy=False)

        with self._lock:
            if e_np.size and e_np.min() == e_np.max():     # fast path: all events in one env (num_envs=1)
                self._events[int(e_np[0])][camera_name].append(
                    (x_np, y_np, t_np, p_np, q_np))
            else:
                for e in np.unique(e_np):                   # group by env, still array-wise (no per-event loop)
                    m = e_np == e
                    self._events[int(e)][camera_name].append(
                        (x_np[m], y_np[m], t_np[m], p_np[m], q_np[m]))

    def flush_episode(self, env_id: int, episode_idx: int, time_origin_s=None,
                      metadata=None):
        """Flush all cameras for one environment to an HDF5 event file.

        ``time_origin_s`` is the absolute Isaac simulation time of the episode's
        first source frame. Event timestamps remain losslessly absolute; the
        metadata lets evaluators and dataset readers convert them to relative
        episode time without guessing from the first threshold crossing.
        """
        with self._lock:
            if env_id not in self._events or not self._events[env_id]:
                return

            # Pop the environment's data out of the active buffer
            env_data = self._events.pop(env_id)

        filename = os.path.join(self.output_dir, f"env{env_id}_ep{episode_idx}.h5")

        counts = {}
        with h5py.File(filename, "w") as f:
            if time_origin_s is not None:
                f.attrs["event_time_origin_s"] = float(time_origin_s)
            for key, value in (metadata or {}).items():
                f.attrs[str(key)] = value
            for cam_name, chunks in env_data.items():
                if not chunks: continue

                # Create a group for each camera in this environment
                grp = f.create_group(cam_name)

                # Each chunk is (x_arr, y_arr, t_arr, p_arr, q_arr).
                xs = np.concatenate([c[0] for c in chunks]).astype(np.uint16)
                ys = np.concatenate([c[1] for c in chunks]).astype(np.uint16)
                ts = np.concatenate([c[2] for c in chunks]).astype(np.float64)
                ps = np.concatenate([c[3] for c in chunks]).astype(np.int8)
                qs = np.concatenate([c[4] for c in chunks]).astype(np.float16)
                counts[cam_name] = int(xs.shape[0])

                # Write each dataset in blocks rather than one monolithic
                # create_dataset(data=...): a single ~100M-event compressed write
                # can corrupt a filter chunk; block writes through the chunk cache
                # are the robust path for very large event streams.
                self._write_blocked(grp, "x", xs)
                self._write_blocked(grp, "y", ys)
                self._write_blocked(grp, "t", ts)
                self._write_blocked(grp, "p", ps)
                self._write_blocked(grp, "q", qs)

        summary = ", ".join(f"{k}={v:,}" for k, v in counts.items())
        print(f"[Recorder] Saved Env {env_id} (Ep {episode_idx}) w/ {len(env_data)} cameras "
              f"[{summary}] comp={self.compression} -> {filename}")

    def _write_blocked(self, grp, name, arr, block: int = 8_000_000):
        """Create a 1-D dataset and fill it in ``block``-sized slices."""
        n = int(arr.shape[0])
        if n == 0:
            grp.create_dataset(name, shape=(0,), dtype=arr.dtype)
            return
        # compression requires chunked storage; uncompressed stays contiguous
        chunks = True if self.compression else None
        ds = grp.create_dataset(name, shape=(n,), dtype=arr.dtype,
                                compression=self.compression, chunks=chunks)
        for i in range(0, n, block):
            ds[i:i + block] = arr[i:i + block]
