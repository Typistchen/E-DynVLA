
import logging
import math
import os
import shutil
import time

import diffusers.optimization
import torch

import core.test
import utils.average_meter
import utils.datasets
import utils.distributed
import utils.helpers
import utils.summary_writer


def train(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    rank, local_rank = utils.distributed.get_rank(), utils.distributed.get_local_rank()

    # Set up datasets
    # train_dataset = lerobot.datasets.lerobot_dataset.LeRobotDataset(
    train_dataset = utils.datasets.get_dataset(
        cfg.DATASET.NAME,
        split="train",
        pin_memory=cfg.DATASET.PIN_MEMORY,
        delta_action=cfg.DATASET.USE_DELTA_ACTION,
        required_features=cfg.DATASET.REQUIRED_FEATURES,
        image_transforms=utils.datasets.ImageTransforms(
            cfg.DATASET.IMG_SIZE, cfg.TRAIN.IMAGE_TRANSFORMS
        ),
        delta_timestamps=utils.helpers.get_delta_timestamps(
            cfg.POLICY, cfg.DATASET.DELTA_TIMESTAMPS
        ),
        event_manifest=cfg.DATASET.get("EVENT_MANIFEST"),
        event_root=cfg.DATASET.get("EVENT_ROOT"),
        event_history_bins=cfg.POLICY.get("EVENT_HISTORY_BINS", 8),
        event_bin_ms=cfg.DATASET.get("EVENT_BIN_MS", 10.0),
        event_output_size=cfg.DATASET.get("EVENT_OUTPUT_SIZE", (96, 128)),
        event_future_steps=cfg.POLICY.get("WAM_FUTURE_STEPS", 10),
        event_future_grid_size=cfg.POLICY.get("WAM_GRID_SIZE", (12, 16)),
        action_horizon=cfg.POLICY.get("CHUNK_SIZE", 20),
        rotation_format=cfg.DATASET.get("ROTATION_FORMAT", "euler"),
        **utils.datasets.get_edv_dataset_kwargs(cfg),
    )
    test_dataset = utils.datasets.get_dataset(
        cfg.DATASET.NAME,
        split="test",
        pin_memory=cfg.DATASET.PIN_MEMORY,
        delta_action=cfg.DATASET.USE_DELTA_ACTION,
        required_features=cfg.DATASET.REQUIRED_FEATURES,
        image_transforms=utils.datasets.ImageTransforms(cfg.DATASET.IMG_SIZE),
        delta_timestamps=utils.helpers.get_delta_timestamps(
            cfg.POLICY, cfg.DATASET.DELTA_TIMESTAMPS
        ),
        event_manifest=cfg.DATASET.get("EVENT_MANIFEST"),
        event_root=cfg.DATASET.get("EVENT_ROOT"),
        event_history_bins=cfg.POLICY.get("EVENT_HISTORY_BINS", 8),
        event_bin_ms=cfg.DATASET.get("EVENT_BIN_MS", 10.0),
        event_output_size=cfg.DATASET.get("EVENT_OUTPUT_SIZE", (96, 128)),
        event_future_steps=cfg.POLICY.get("WAM_FUTURE_STEPS", 10),
        event_future_grid_size=cfg.POLICY.get("WAM_GRID_SIZE", (12, 16)),
        action_horizon=cfg.POLICY.get("CHUNK_SIZE", 20),
        rotation_format=cfg.DATASET.get("ROTATION_FORMAT", "euler"),
        **utils.datasets.get_edv_dataset_kwargs(cfg),
    )
    train_sampler = None
    test_sampler = None
    if torch.cuda.is_available():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, rank=rank, shuffle=True, drop_last=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, rank=rank, shuffle=False, drop_last=True
        )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.N_WORKERS,
        pin_memory=cfg.DATASET.PIN_MEMORY,
        sampler=train_sampler,
        persistent_workers=True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=min(2, cfg.CONST.N_WORKERS),
        pin_memory=cfg.DATASET.PIN_MEMORY,
        sampler=test_sampler,
        persistent_workers=False,
    )

    # A frozen randomly initialized backbone cannot learn. Fail before model
    # construction so a missing command-line checkpoint is never overlooked.
    checkpoint = cfg.CONST.get("CKPT") or cfg.POLICY.get("CHECKPOINT")
    if checkpoint and not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not checkpoint and any(
        cfg.POLICY.get(name, False)
        for name in ("FREEZE_VISION_MODEL", "FREEZE_CONNECTOR", "FREEZE_TEXT_MODEL")
    ):
        raise ValueError(
            "A pretrained checkpoint is required when the vision/connector/text "
            "backbone is frozen. Pass --ckpt or set POLICY.CHECKPOINT."
        )

    # Set up the policy
    policy = utils.helpers.get_policy(
        cfg.POLICY,
        train_dataset.meta,
        cfg.DATASET.IMG_SIZE,
        cfg.DATASET.REQUIRED_FEATURES,
    )
    if utils.distributed.is_local_master():
        logging.info(
            "Using policy: %s with config %s" % (cfg.POLICY.TYPE, policy.config)
        )
        logging.info(
            "#Parameters: %s/%s"
            % (
                utils.helpers.get_formatted_big_number(
                    utils.helpers.get_n_parameters(policy, trainable_only=True)
                ),
                utils.helpers.get_formatted_big_number(
                    utils.helpers.get_n_parameters(policy, trainable_only=False)
                ),
            )
        )

    # ``-p`` initializes E-DynVLA from a base policy checkpoint.  Its saved
    # epoch belongs to the source training run and must not skip E-DynVLA
    # epochs (a real resume would also need optimizer/scheduler state).
    init_epoch = 0
    if checkpoint:
        cfg.CONST.CKPT = checkpoint
        logging.info("Loading pretrained model from %s ..." % checkpoint)
        # Save the normalizers to enable migration to the new datasets
        normalizers = {
            n: getattr(policy, n)
            for n in [
                "normalize_inputs",
                "normalize_targets",
                "unnormalize_outputs",
            ]
        }
        policy.config.device = "cuda:%d" % local_rank
        policy = policy.from_pretrained(checkpoint, config=policy.config)
        for k, v in normalizers.items():
            setattr(policy, k, v)

    if torch.cuda.is_available():
        policy = torch.nn.parallel.DistributedDataParallel(
            policy.to(local_rank),
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

    # Set up the optimizer
    n_batches = len(train_data_loader)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=cfg.TRAIN.OPTIMIZER.LR,
        eps=cfg.TRAIN.OPTIMIZER.EPS,
        weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        betas=cfg.TRAIN.OPTIMIZER.BETAS,
    )
    grad_accum_steps = int(cfg.TRAIN.GRAD_ACCUM_STEPS)
    if grad_accum_steps < 1:
        raise ValueError("TRAIN.GRAD_ACCUM_STEPS must be positive")
    optimizer_steps_per_epoch = math.ceil(n_batches / grad_accum_steps)
    lr_scheduler = diffusers.optimization.get_scheduler(
        name=cfg.TRAIN.LR_SCHEDULER.NAME,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.LR_SCHEDULER.N_WARMUP_STEPS,
        num_training_steps=cfg.TRAIN.N_EPOCHS * optimizer_steps_per_epoch,
    )

    # Set up folders for logs, snapshot and checkpoints
    if utils.distributed.is_master():
        output_dir = os.path.join(cfg.DIR.OUTPUT, "%s", cfg.CONST.EXP_NAME)
        cfg.DIR.CHECKPOINTS = output_dir % "checkpoints"
        cfg.DIR.LOGS = output_dir % "logs"
        os.makedirs(cfg.DIR.CHECKPOINTS, exist_ok=True)
        # Summary writer
        tb_writer = utils.summary_writer.SummaryWriter(cfg)
        # Log current config
        tb_writer.add_config(cfg.DATASET)
        tb_writer.add_config(cfg.POLICY)
        tb_writer.add_config(cfg.TRAIN)

    for epoch_idx in range(init_epoch, cfg.TRAIN.N_EPOCHS):
        epoch_start_time = time.perf_counter()
        batch_time = utils.average_meter.AverageMeter()
        data_time = utils.average_meter.AverageMeter()
        train_losses = utils.average_meter.AverageMeter()
        component_losses = {}
        # Randomize the DistributedSampler
        if train_sampler:
            train_sampler.set_epoch(epoch_idx)

        # Training loop
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        batch_end_time = time.perf_counter()
        for batch_idx, batch in enumerate(train_data_loader):
            n_itr = epoch_idx * n_batches + batch_idx
            data_time.update(time.perf_counter() - batch_end_time)
            batch = {
                k: (
                    v.to(policy.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in batch.items()
            }
            # Fix: Remove the additional dimension for task
            if isinstance(batch["task"], list) and isinstance(
                batch["task"][0], (tuple, list)
            ):
                batch["task"] = batch["task"][0]

            loss, loss_dict = policy.forward(batch)
            for name in ("action_loss", "wam_loss", "wam_rgb_loss", "wam_event_loss"):
                if name in loss_dict:
                    component_losses.setdefault(
                        name, utils.average_meter.AverageMeter()
                    ).update(loss_dict[name])
            # The final accumulation group may contain fewer than
            # ``grad_accum_steps`` batches. Scale by its actual size so the
            # last optimizer update has the same magnitude as the others.
            group_start = (batch_idx // grad_accum_steps) * grad_accum_steps
            group_size = min(grad_accum_steps, n_batches - group_start)
            (loss / group_size).backward()
            should_step = (
                (batch_idx + 1) % grad_accum_steps == 0
                or batch_idx + 1 == n_batches
            )
            if should_step:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_losses.update(loss.item())
            batch_time.update(time.perf_counter() - batch_end_time)
            batch_end_time = time.perf_counter()
            if utils.distributed.is_master():
                batch_scalars = {"Loss/Batch": train_losses.val()}
                batch_scalars.update(
                    {
                        f"Loss/{name}/Batch": meter.val()
                        for name, meter in component_losses.items()
                    }
                )
                for name, value in loss_dict.items():
                    if name.startswith("wam_") and name not in component_losses:
                        batch_scalars[f"WAM/{name.removeprefix('wam_')}"] = value
                tb_writer.add_scalars(batch_scalars, n_itr)
                # Save the model checkpoint every few batches
                if (
                    cfg.TRAIN.CKPT_SAVE_FREQ.BATCH != 0
                    and batch_idx % cfg.TRAIN.CKPT_SAVE_FREQ.BATCH == 0
                ):
                    logging.info("Saving checkpoint to %s ..." % cfg.DIR.CHECKPOINTS)
                    utils.helpers.save_checkpoint(
                        cfg, policy, cfg.DIR.CHECKPOINTS, epoch_idx + 1
                    )

            if utils.distributed.is_local_master():
                logging.info(
                    "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %.4f"
                    % (
                        epoch_idx + 1,
                        cfg.TRAIN.N_EPOCHS,
                        batch_idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        train_losses.val(),
                    )
                )

        epoch_end_time = time.perf_counter()
        if utils.distributed.is_master():
            epoch_scalars = {"Loss/Epoch/Train": train_losses.avg()}
            epoch_scalars.update(
                {
                    f"Loss/{name}/Epoch": meter.avg()
                    for name, meter in component_losses.items()
                }
            )
            tb_writer.add_scalars(epoch_scalars, epoch_idx)

        if utils.distributed.is_local_master():
            logging.info(
                "[Epoch %d/%d] EpochTime = %.3f (s) Losses = %.4f"
                % (
                    epoch_idx + 1,
                    cfg.TRAIN.N_EPOCHS,
                    epoch_end_time - epoch_start_time,
                    train_losses.avg(),
                )
            )

        # Evaluate the current model
        test_losses = core.test(
            cfg,
            test_data_loader=test_data_loader,
            policy=policy,
        )
        if utils.distributed.is_master():
            tb_writer.add_scalars({"Loss/Epoch/Test": test_losses.avg()}, epoch_idx)

        # Save the model checkpoint
        if utils.distributed.is_master():
            logging.info("Saving checkpoint to %s ..." % cfg.DIR.CHECKPOINTS)
            utils.helpers.save_checkpoint(
                cfg, policy, cfg.DIR.CHECKPOINTS, epoch_idx + 1
            )
            if epoch_idx % cfg.TRAIN.CKPT_SAVE_FREQ.EPOCH == 0:
                shutil.copy(
                    os.path.join(cfg.DIR.CHECKPOINTS, "model.safetensors"),
                    os.path.join(
                        cfg.DIR.CHECKPOINTS, "model.epoch%04d.safetensors" % epoch_idx
                    ),
                )
