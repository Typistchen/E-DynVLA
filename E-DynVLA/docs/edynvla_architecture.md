# E-DynVLA event-token architecture

E-DynVLA consumes the DOM RGB observations, robot state, language instruction,
and the confidence-weighted output of the lighting-suppressed ego-motion
separator. It does **not** use semantic segmentation or object velocity as
model inputs.

```text
DOM RGB + language ----------------------> FastViT/SmolLM2 tokens
robot state -----------------------------> state token
v4-hybrid events
  -> ego/light separation
  -> static event history  --\
                                sparse event tokenizer -> event tokens
  -> dynamic event history --/

[RGB, language, static-event, dynamic-event, state] prefix
  -> DynamicVLA flow-matching action expert -> action chunk
  -> Event-WAM auxiliary head              -> future event activity
```

## Why this is not an object detector

The event tokenizer selects active **patches**, not object boxes. No class,
instance ID, or detection target is needed. Static and dynamic streams share
one patch encoder and are distinguished with learned type embeddings. The
result is a global spatio-temporal token sequence suitable for VLM attention.

## Input representation

Each synchronized sample contains two tensors of shape `[T, 2, H, W]`:

- `observation.events.static`: background events consistent with camera ego
  motion, weighted by `q_static * (1 - q_illumination)`;
- `observation.events.dynamic`: residual-motion events, weighted by
  `q_dynamic * (1 - q_illumination)`.

The two polarity channels are OFF and ON. The default history has eight 10 ms
bins at 96 x 128 resolution. The tokenizer keeps eight high-density patches
per bin and modality, adds time/type/coordinate embeddings, aggregates them
with a two-layer Transformer, and projects them to the 768-wide VLM space.

## Event-WAM target

The auxiliary target is ten future 10 ms steps on a 12 x 16 patch grid with
four channels: static-OFF, static-ON, dynamic-OFF, and dynamic-ON. The action
loss remains the existing flow-matching objective. The WAM loss teaches the
shared event tokenizer short-horizon dynamics and can be disabled for a
controlled ablation.

## Dataset contract

Use `scripts/build_edynvla_manifest.py` to index DOM episode HDF5 files and the
corresponding eventized/separated HDF5 files. Simulator-only `segmentation` and
`object_vel` are explicitly marked evaluation-only in the manifest.

`policies.edynvla.data.DOMEventDataset` reads that manifest directly and emits
paired RGB, 7-DoF end-effector state, action chunks, static/dynamic histories,
and future-event supervision. Set `EDYNVLA_DATA_ROOT` (or pass
`dataset_root`) to the external dataset directory; large HDF5 files are not
stored in Git.

The existing ten-demo set is an integration and evaluation set, not enough to
train a language/action model from scratch. Training should use the full DOM
episode corpus and its v4-hybrid eventized counterpart; the Small VLM remains
frozen initially while the event tokenizer, projector, Event-WAM head, and
action expert are optimized.

## Required ablations

1. RGB DynamicVLA baseline.
2. RGB + raw event tokens.
3. RGB + static event tokens.
4. RGB + dynamic event tokens.
5. RGB + static/dynamic event tokens.
6. RGB + static/dynamic event tokens + Event-WAM.
