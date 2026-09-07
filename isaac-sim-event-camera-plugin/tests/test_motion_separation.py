import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from separate_dynamic_static_events import (  # noqa: E402
    AsyncSeparatedEventVoxelizer,
    ReusableSeparatedEventVoxelizer,
    SeparatedEventVoxelRing,
    StreamingMotionSeparator,
    ego_geometry,
    fuse_motion_lighting,
    robust_motion_calibration,
    voxelize_separated_events,
    voxelize_separated_events_torch,
)


def _camera_sequence(n_frames=4, height=20, width=30):
    depth = np.ones((n_frames, height, width), np.float32)
    poses = np.zeros((n_frames, 7), np.float64)
    poses[:, 0] = np.arange(n_frames) * 0.1
    poses[:, 3] = 1.0
    intrinsics = np.repeat(
        np.array(
            [[50.0, 0.0, width / 2], [0.0, 50.0, height / 2], [0.0, 0.0, 1.0]],
            np.float64,
        )[None],
        n_frames,
        axis=0,
    )
    return depth, poses, intrinsics


def test_ego_geometry_identity_keeps_depth_and_zero_flow():
    depth, poses, intrinsics = _camera_sequence(n_frames=2)
    poses[1] = poses[0]
    flow, predicted_depth, valid = ego_geometry(
        depth[0], poses[0], poses[1], intrinsics[0]
    )
    assert np.allclose(flow[valid], 0.0, atol=1e-5)
    assert np.allclose(predicted_depth[valid], 1.0, atol=1e-5)


def test_calibration_uses_dominant_geometry_without_semantics():
    depth, poses, intrinsics = _camera_sequence()
    motion = np.zeros((*depth.shape, 2), np.float32)
    # +X camera translation produces -5 px static flow. Isaac convention in
    # this fixture is offset=1, sign=-1, scale=1.25.
    motion[1:, ..., 0] = 4.0
    calibration = robust_motion_calibration(depth, motion, poses, intrinsics)
    assert calibration["offset"] == 1
    assert calibration["sign"] == -1.0
    assert np.isclose(calibration["scale"], 1.25, atol=1e-3)


def test_lighting_without_geometry_is_rejected_from_dynamic():
    shape = (1, 1)
    valid = np.ones(shape, bool)
    static, dynamic, illumination, unknown = fuse_motion_lighting(
        flow_confidence=np.full(shape, 0.9, np.float32),
        depth_confidence=np.zeros(shape, np.float32),
        photo_confidence=np.ones(shape, np.float32),
        chroma_confidence=np.zeros(shape, np.float32),
        persistent=np.zeros(shape, np.float32),
        valid=valid,
    )
    assert illumination.item() > 0.9
    assert dynamic.item() < 0.5
    assert unknown.item() > 0.9
    assert static.item() < 0.1


def test_depth_supported_motion_survives_brightness_change():
    shape = (1, 1)
    _, dynamic, illumination, _ = fuse_motion_lighting(
        flow_confidence=np.full(shape, 0.9, np.float32),
        depth_confidence=np.full(shape, 0.9, np.float32),
        photo_confidence=np.ones(shape, np.float32),
        chroma_confidence=np.zeros(shape, np.float32),
        persistent=np.zeros(shape, np.float32),
        valid=np.ones(shape, bool),
    )
    assert illumination.item() < 0.11
    assert dynamic.item() > 0.8


def test_streaming_cached_geometry_matches_reference():
    depth, poses, intrinsics = _camera_sequence(n_frames=2)
    calibration = {"offset": 1, "sign": -1.0, "scale": 1.0}
    separator = StreamingMotionSeparator(calibration)
    expected_flow, expected_depth, expected_valid = ego_geometry(
        depth[0], poses[0], poses[1], intrinsics[0]
    )
    flow, predicted_depth, valid = separator._ego_geometry(
        depth[0], poses[0], poses[1], intrinsics[0]
    )
    np.testing.assert_allclose(flow, expected_flow, atol=1e-5)
    np.testing.assert_allclose(predicted_depth, expected_depth, atol=1e-5)
    np.testing.assert_array_equal(valid, expected_valid)


def test_streaming_separator_reuses_geometry_and_returns_confidence_maps():
    depth, poses, intrinsics = _camera_sequence(n_frames=2)
    poses[1] = poses[0]
    height, width = depth.shape[1:]
    rgb = np.full((height, width, 3), 100, np.uint8)
    motion = np.zeros((height, width, 2), np.float32)
    separator = StreamingMotionSeparator(
        {"offset": 1, "sign": -1.0, "scale": 1.0}
    )

    result = separator.step(
        rgb,
        depth[0],
        poses[0],
        rgb,
        depth[1],
        poses[1],
        intrinsics[0],
        motion,
    )
    cached_grid = separator._xx
    second = separator.step(
        rgb,
        depth[0],
        poses[0],
        rgb,
        depth[1],
        poses[1],
        intrinsics[0],
        motion,
    )

    assert separator._xx is cached_grid
    for key in (
        "q_motion",
        "q_static",
        "q_dynamic",
        "q_illumination",
        "q_unknown",
    ):
        assert result[key].shape == (height, width)
        assert result[key].dtype == np.float32
        assert np.isfinite(result[key]).all()
        assert np.all((result[key] >= 0.0) & (result[key] <= 1.0))
        assert second[key].shape == result[key].shape
    assert result["valid"].dtype == bool
    assert result["q_static"].mean() > result["q_dynamic"].mean()


def test_fused_separated_voxelizer_uses_polarity_time_and_illumination():
    confidence = {
        "q_static": np.ones((4, 4), np.float32),
        "q_dynamic": np.full((4, 4), 0.5, np.float32),
        "q_illumination": np.zeros((4, 4), np.float32),
    }
    # The last event is fully rejected by the illumination gate.
    confidence["q_illumination"][3, 3] = 1.0
    static, dynamic = voxelize_separated_events(
        x=np.array([0, 1, 2, 3], np.uint16),
        y=np.array([0, 1, 2, 3], np.uint16),
        timestamps=np.array([0.001, 0.002, 0.011, 0.012]),
        polarity=np.array([-1, 1, -1, 1], np.int8),
        confidence_maps=confidence,
        start_time=0.0,
        num_bins=2,
        bin_seconds=0.01,
        output_size=(2, 2),
        clip_count=0.0,
    )
    assert static.shape == (2, 2, 2, 2)
    assert dynamic.shape == static.shape
    assert np.isclose(static.sum(), 3.0)
    assert np.isclose(dynamic.sum(), 1.5)
    assert static[0, 0, 0, 0] == 1.0
    assert static[0, 1, 0, 0] == 1.0
    assert static[1, 0, 1, 1] == 1.0
    assert static[1, 1, 1, 1] == 0.0


def test_torch_voxelizer_matches_numpy_on_cpu():
    confidence = {
        "q_static": np.full((4, 4), 0.75, np.float32),
        "q_dynamic": np.full((4, 4), 0.25, np.float32),
        "q_illumination": np.full((4, 4), 0.1, np.float32),
    }
    args = dict(
        x=np.array([0, 1, 2, 3], np.uint16),
        y=np.array([0, 1, 2, 3], np.uint16),
        timestamps=np.array([0.001, 0.002, 0.011, 0.012]),
        polarity=np.array([-1, 1, -1, 1], np.int8),
        confidence_maps=confidence,
        start_time=0.0,
        num_bins=2,
        bin_seconds=0.01,
        output_size=(2, 2),
    )
    expected = voxelize_separated_events(**args)
    actual = voxelize_separated_events_torch(**args, device="cpu")
    np.testing.assert_allclose(actual[0].numpy(), expected[0], atol=1e-6)
    np.testing.assert_allclose(actual[1].numpy(), expected[1], atol=1e-6)


def test_mirrored_voxel_ring_returns_contiguous_chronological_history():
    ring = SeparatedEventVoxelRing(history_bins=4, output_size=(2, 2))
    for value in range(1, 7):
        static = np.full((1, 2, 2, 2), value, np.float32)
        dynamic = -static
        history_static, history_dynamic = ring.append(static, dynamic)
    assert history_static.flags.c_contiguous
    assert history_dynamic.flags.c_contiguous
    np.testing.assert_array_equal(history_static[:, 0, 0, 0], [3, 4, 5, 6])
    np.testing.assert_array_equal(history_dynamic[:, 0, 0, 0], [-3, -4, -5, -6])


def test_async_voxelizer_matches_synchronous_result():
    confidence = {
        "q_static": np.ones((2, 2), np.float32),
        "q_dynamic": np.full((2, 2), 0.5, np.float32),
        "q_illumination": np.zeros((2, 2), np.float32),
    }
    args = dict(
        x=np.array([0, 1], np.uint16),
        y=np.array([0, 1], np.uint16),
        timestamps=np.array([0.001, 0.011]),
        polarity=np.array([-1, 1], np.int8),
        confidence_maps=confidence,
        start_time=0.0,
        num_bins=2,
        bin_seconds=0.01,
        output_size=(2, 2),
    )
    expected = voxelize_separated_events(**args)
    with AsyncSeparatedEventVoxelizer() as voxelizer:
        voxelizer.submit(**args)
        actual = voxelizer.result()
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_reusable_voxelizer_matches_generic_and_reuses_capacity():
    confidence = {
        "q_static": np.full((4, 4), 0.7, np.float32),
        "q_dynamic": np.full((4, 4), 0.3, np.float32),
        "q_illumination": np.full((4, 4), 0.2, np.float32),
    }
    args = dict(
        x=np.array([0, 1, 2, 3], np.uint16),
        y=np.array([0, 1, 2, 3], np.uint16),
        timestamps=np.array([0.001, 0.002, 0.011, 0.012]),
        polarity=np.array([-1, 1, -1, 1], np.int8),
        confidence_maps=confidence,
        start_time=0.0,
        num_bins=2,
        bin_seconds=0.01,
        output_size=(2, 2),
    )
    expected = voxelize_separated_events(**args)
    reusable = ReusableSeparatedEventVoxelizer(
        source_size=(4, 4), output_size=(2, 2)
    )
    call_args = dict(args)
    call_args.pop("output_size")
    actual = reusable.voxelize(**call_args)
    capacity = reusable.capacity
    second = reusable.voxelize(**call_args)
    assert reusable.capacity == capacity
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-6)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-6)
    np.testing.assert_array_equal(second[0], actual[0])
    np.testing.assert_array_equal(second[1], actual[1])


def test_reusable_voxelizer_does_not_overflow_uint16_pixel_coordinates():
    confidence = {
        "q_static": np.ones((360, 480), np.float32),
        "q_dynamic": np.full((360, 480), 0.5, np.float32),
        "q_illumination": np.zeros((360, 480), np.float32),
    }
    args = dict(
        x=np.array([479], np.uint16),
        y=np.array([359], np.uint16),
        timestamps=np.array([0.001]),
        polarity=np.array([1], np.int8),
        confidence_maps=confidence,
        start_time=0.0,
        num_bins=1,
        bin_seconds=0.01,
    )
    expected = voxelize_separated_events(**args)
    actual = ReusableSeparatedEventVoxelizer().voxelize(**args)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
