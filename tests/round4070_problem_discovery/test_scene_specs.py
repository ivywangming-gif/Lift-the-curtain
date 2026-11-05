from research.round4070_problem_discovery.code.scene_specs import scene_specs


def test_multi_sheet_scene_geometry_is_explicit_and_bounded() -> None:
    scenes = scene_specs()
    assert set(scenes) == {
        "split_free", "split_left_limited", "split_right_limited", "strips_3_overlap",
        "split_free_layered", "split_left_limited_layered", "split_right_limited_layered",
        "strips_3_overlap_layered",
        "split_free_layered24", "split_left_limited_layered24", "split_right_limited_layered24",
        "strips_3_overlap_layered40",
    }
    for scene in scenes.values():
        scene.validate()
        assert len(scene.sheets) >= 2
        for sheet in scene.sheets:
            config = sheet.curtain_config(__import__("lift_curtain.envs.task_scale_scene", fromlist=["FormalTaskSceneConfig"]).FormalTaskSceneConfig())
            assert abs((config.support_z_m - config.height_m) - 0.02) < 1.0e-9
            assert config.yaw_rad > 1.5


def test_split_limit_is_mirrored_without_changing_sheet_geometry() -> None:
    scenes = scene_specs()
    left = scenes["split_left_limited"]
    right = scenes["split_right_limited"]
    assert [(s.center_y_m, s.width_m, s.columns, s.center_x_offset_m) for s in left.sheets] == [(s.center_y_m, s.width_m, s.columns, s.center_x_offset_m) for s in right.sheets]
    assert [s.bottom_anchor_mode for s in left.sheets] == ["none", "right"]
    assert [s.bottom_anchor_mode for s in right.sheets] == ["left", "none"]


def test_layered_scenes_preserve_projection_and_add_only_depth_spacing() -> None:
    scenes = scene_specs()
    flat = scenes["split_free"]
    layered = scenes["split_free_layered"]
    assert [(s.center_y_m, s.width_m, s.columns, s.bottom_anchor_mode) for s in flat.sheets] == [
        (s.center_y_m, s.width_m, s.columns, s.bottom_anchor_mode) for s in layered.sheets
    ]
    assert [s.center_x_offset_m for s in layered.sheets] == [-0.004, 0.004]
    assert [s.center_x_offset_m for s in scenes["split_free_layered24"].sheets] == [-0.012, 0.012]
