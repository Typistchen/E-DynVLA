# E-DynVLA

E-DynVLA is a compact research repository for dynamic manipulation with RGB,
robot state, language, and event-camera observations. The repository contains
two maintained modules:

- [`V2E-VLA/`](V2E-VLA/): v4-hybrid RGB/HDR-to-event generation, event I/O,
  visualization, evaluation, and lighting-aware motion separation.
- [`E-DynVLA/`](E-DynVLA/): DOM/Isaac Lab data generation, reproducible EDV
  packaging, event tokenization, Event-WAM, training, and inference.

[`benchmark/`](benchmark/) contains controlled v2/v3/v4 and motion-separation
evaluation scripts. Generated datasets, assets, environments, and checkpoints
are intentionally stored outside Git.

## Data pipeline

```text
DOM + Isaac Lab
  -> three RGB views + aligned robot/action data
  -> V2E-VLA v4-hybrid raw events
  -> lighting-suppressed static/dynamic event streams
  -> sparse event tokens + RGB/language/state
  -> E-DynVLA action prediction (+ optional Event-WAM)
```

The generated dataset is organized as `success/sample_xxxxxx/` and
`failure/sample_xxxxxx/`. Each sample contains observation-aligned Parquet,
three RGB MP4 files, three AEDAT4 event streams, and `reproduction.json`.

## Quick links

- [Architecture](E-DynVLA/docs/edynvla_architecture.md)
- [Event generation](E-DynVLA/docs/event_camera.md)
- [V2E-VLA module](V2E-VLA/README.md)
- [Controlled benchmarks](benchmark/README.md)

## Upstream components

Only the DynamicVLA and EVIS components required by this project are retained.
They have been integrated into the two modules above; the original project
layouts, examples, media, and unrelated utilities are not mirrored here.
Licenses and attribution are preserved in the module license files and
[`THIRD_PARTY.md`](THIRD_PARTY.md).
