# BIMANUAL_V1 Dynamic Scene Build Report

Status: OFFLINE_DYNAMIC_BUILDER_PREPARATION_ONLY.

- Candidate scenes: 56.
- REQUIRES_PHYSICS_VALIDATION: 53.
- NOT_BUILT: 3 (all are missing external real-material assets).
- ASSET_READY: 0; no native USD was generated because Kit launch is prohibited under the current pause.
- Builder-ready components: 53; unbuildable material dependencies: 3.

## Component coverage

- Fabric curtain: reusable surface-deformable mesh/attachment recipe for every non-control surface scene.
- Rigid obstacle: hidden wall and rigid-board cuboid recipe with metric dimensions and collision schema mapping.
- PVC/mesh: explicit proxy representation and calibration warning; real material rows are not silently substituted.
- Hinged door: four reusable UsdPhysics.RevoluteJoint plans with hinge axis, gap, resistance, pushability and label scope.
- Hand collider: referenced certification remains non-admitting; no plan upgrades B10/V6 evidence.

## Per-scene status

### REQUIRES_PHYSICS_VALIDATION
- BI_A01_BASIC_CLOTH
- BI_A02_RIGID_BOARD
- BI_A03_EMPTY_CONTROL
- BI_B_WALL_02CM
- BI_B_WALL_04CM
- BI_B_WALL_06CM
- BI_B_WALL_08CM
- BI_B_WALL_NOWALL
- BI_C_BOTH_BOTTOM_CORNERS
- BI_C_FOUR_CORNERS
- BI_C_FULL_PERIMETER
- BI_C_LEFT_BOTTOM_CORNER
- BI_C_LEFT_EDGE_FIXED
- BI_C_RIGHT_BOTTOM_CORNER
- BI_C_RIGHT_EDGE_FIXED
- BI_C_THREE_MISSING_BOTTOM_LEFT
- BI_C_THREE_MISSING_BOTTOM_RIGHT
- BI_C_THREE_MISSING_TOP_LEFT
- BI_C_THREE_MISSING_TOP_RIGHT
- BI_C_TOP_PLUS_BOTTOM_CORNERS
- BI_D_BEND_HIGH
- BI_D_DAMPING_HIGH
- BI_D_DENSITY_HIGH
- BI_D_DENSITY_ORDINARY
- BI_D_DENSITY_VERY_HIGH
- BI_D_FRICTION_HIGH
- BI_D_HEM_MASS
- BI_D_INITIAL_RELAXED
- BI_D_INITIAL_TENSIONED
- BI_D_PLEAT_GENTLE
- BI_D_PLEAT_LARGE
- BI_D_STRETCH_HIGH
- BI_D_THICKNESS_HIGH
- BI_E_CLOTH_PROXY
- BI_E_DENSE_MESH_PROXY
- BI_E_DOUBLE_CENTER_OPENING
- BI_E_LEFT_OPEN
- BI_E_OPEN_MESH_PROXY
- BI_E_PVC_INDEPENDENT_PROXY
- BI_E_PVC_OVERLAP_PROXY
- BI_E_PVC_SPACING_VARIANT
- BI_E_PVC_THICKNESS_VARIANT
- BI_E_PVC_WIDTH_VARIANT
- BI_E_RIGHT_OPEN
- BI_F_CENTER_SPLIT_MOVABLE
- BI_F_INDEPENDENT_RINGS
- BI_F_LEFT_OPEN_MOVABLE
- BI_F_RIGHT_OPEN_MOVABLE
- BI_F_TOP_RAIL
- BI_G_DOOR_HIGH
- BI_G_DOOR_MEDIUM
- BI_G_DOOR_NONPUSHABLE
- BI_G_DOOR_ORDINARY

### NOT_BUILT
- BI_E_REAL_CLOTH
- BI_E_REAL_MESH
- BI_E_REAL_PVC

## Interpretation

REQUIRES_PHYSICS_VALIDATION means the dynamic recipe is complete and mapped to the installed native API, but no USD stage was authored and no contact/release/recovery/penetration result exists. It is not a physics pass.

NOT_BUILT means a required real-material source is absent. The corresponding proxy may be represented separately, but it cannot stand in for the requested material condition.

The complete machine-readable per-scene table is BIMANUAL_DYNAMIC_SCENE_READINESS.csv; each scene plan is under bimanual_v1_assets/dynamic_scene_plans/.
