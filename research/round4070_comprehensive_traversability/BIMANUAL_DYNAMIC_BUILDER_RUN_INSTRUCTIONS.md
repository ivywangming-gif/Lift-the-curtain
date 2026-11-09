# BIMANUAL_V1 Dynamic Builder Run Instructions

Status: OFFLINE PREPARATION ONLY. This command does not launch Kit, Isaac, a physics gate, or G1. It requires both pause flags and fails closed if either flag is absent.

    Set-Location -LiteralPath 'C:\Robotics\Lift-the-curtain'
    & 'C:\conda_envs\isaaclab\python.exe' scripts/round4070_g/prepare_bimanual_dynamic_scenes.py

The command reads the existing 56 candidate scene assets, writes one deterministic dynamic plan per candidate, appends readiness columns to the existing BIMANUAL_V1 matrix, and updates the Stage 2 checkpoint. It does not recreate scene IDs, reset the campaign manifest, or overwrite raw history.

Generated plans are under research/round4070_comprehensive_traversability/bimanual_v1_assets/dynamic_scene_plans/. Native authoring is exposed only through the explicit BimanualSceneComposer.build_native(stage, asset) API and is not called by this offline command.

Formal physics or G1 admission still requires the unchanged B10/V6 safety gates, exact native hand collider admission, representative stop/retract physics-only validation, independent safety approval, and a material-aware budget decision.
