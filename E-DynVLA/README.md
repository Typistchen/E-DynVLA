# E-DynVLA module

This module contains the VLA side of the project:

- `policies/edynvla/`: sparse event tokenizer, DOM/event adapter, and an
  action-conditioned WAM that predicts future RGB patches and future events;
- `policies/dynamicvla/`: retained DynamicVLA backbone required by E-DynVLA;
- `simulations/`: retained and adapted DOM/Isaac Lab simulation path;
- `scripts/`: reproducible EDV generation, packaging, validation, and status;
- `configs/edynvla.yaml`: main E-DynVLA training configuration.

The module expects the sibling [`../V2E-VLA`](../V2E-VLA/) package for event
generation. The import package remains `dvs_gen` for compatibility with the
Isaac camera integration.

## Generate EDV samples

On the configured Isaac Lab server:

```bash
bash scripts/generate_edv_samples.sh 0 5 cuda:2
```

Generate in parallel until the dataset reaches a target size:

```bash
bash scripts/generate_edv_dataset_to_size.sh 500 cuda:2 cuda:3
```

Large assets, datasets, environments, and checkpoints are external inputs and
are not committed to this repository.

Before distributed training, build the versioned event-separation cache once
in a single process. This avoids several DataLoader workers decoding and
separating the same large AEDAT4 episode concurrently:

```bash
python scripts/precompute_edv_event_cache.py \
  --dataset-root /path/to/EDV_Support \
  --v2e-root ../V2E-VLA
```

Training with the frozen backbone requires a pretrained DynamicVLA checkpoint;
pass it with `--ckpt` or set `POLICY.CHECKPOINT`.

See [docs/edynvla_architecture.md](docs/edynvla_architecture.md) for the model
contract and [../THIRD_PARTY.md](../THIRD_PARTY.md) for retained upstream code.
