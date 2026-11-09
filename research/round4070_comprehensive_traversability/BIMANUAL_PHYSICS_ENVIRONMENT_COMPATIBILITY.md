# BIMANUAL_V1 Physics Environment Compatibility

Status: OFFLINE_CAPABILITY_AUDIT_ONLY. No Kit, Isaac, physics gate, or G1 process was started.

## Installed environment

- Windows: Windows 10 Enterprise; DisplayVersion 25H2; build 26200; UBR 9711
- Python: 3.11.15 (C:\conda_envs\isaaclab\python.exe)
- Isaac Sim package: 5.1.0.0
- Isaac Lab package: 0.54.4
- PyTorch: 2.7.0+cu128 (bundled CUDA runtime 12.8)
- CUDA available: True; device: NVIDIA GeForce RTX 4070 Laptop GPU; VRAM reported by torch: 7.996 GiB
- nvidia-smi: NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB, 610.60
- PhysX extensions:
- 107.3.26: C:\conda_envs\isaaclab\Lib\site-packages\isaacsim\extscache\omni.physics.physx-107.3.26+107.3.3.wx64.r.cp311.u353 (Physics PhysX Backend.)
- 107.3.26: C:\conda_envs\isaaclab\Lib\site-packages\isaacsim\extscache\omni.physx-107.3.26+107.3.3.wx64.r.cp311.u353 (PhysX 5 integration code.)
- PhysX backend extension version: 107.3.26

## Compatibility judgement

- The installed source and PhysX extension expose the Isaac Sim 5.1 surface-deformable path: UsdGeom.Mesh, set_physics_surface_deformable_body, add_surface_deformable_material, automatic deformable attachments, and UsdPhysics.CollisionAPI.
- A G1 rigid articulation and a surface-deformable curtain can be authored in one Kit/PhysX stage in principle. The existing runtime builder uses the same stage for deformable sheets, rigid supports, and barriers.
- This is API/source compatibility evidence only. Contact stability, attachment resolution, hinged-door limits, native hand binding, and coordinated bimanual behavior still require a permitted physics-only validation.
- The RTX 4070 Laptop GPU has enough reported memory for staged offline work and one-scene validation; this does not justify concurrent execution of 56 scenes or the registered seed matrix.
- No environment upgrade was performed.

## Safety boundary

B10/V6 remains NO-GO. The native hand certification file still reports no formal exact-collider admission. Dynamic plans therefore remain unverified and cannot authorize a launch.
