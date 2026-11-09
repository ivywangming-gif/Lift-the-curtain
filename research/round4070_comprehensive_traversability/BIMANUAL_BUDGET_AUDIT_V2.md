# BIMANUAL_V1 Budget Audit V2

Status: OFFLINE AUDIT; no budget or manifest was changed.

## Relationship of the registered numbers

- 56 candidate rows are named benchmark cells. Their conservative target is 1820 valid synchronized bimanual episodes.
- The rows request 595 independent parameter combinations. At three physics seeds this is 1785 physics seed checks.
- A physics seed check is not a G1 episode, and left/right hands are not two samples. A passed physics configuration can support multiple episodes only after the protocol explicitly permits that reuse.

## Duplicate audit

- Existing structural physical_config_id count: 50.
- Material-aware execution strata: 53.
- Material-aware lower bound after exact duplicate removal: 1710 valid episodes, 560 combinations, 1680 seed checks.
- The tempting structural-ID lower bound is 1620 episodes and 530 combinations, but it is not material-safe because the current hash excludes material_fidelity.
- Duplicate groups that are safe to share only within the same material fidelity:
- BI_A01_BASIC_CLOTH, BI_B_WALL_NOWALL, BI_D_DENSITY_ORDINARY, BI_E_CLOTH_PROXY: same physical/material execution stratum

- All structural duplicate groups, including unsafe material aliases:
- BI_A01_BASIC_CLOTH, BI_B_WALL_NOWALL, BI_D_DENSITY_ORDINARY, BI_E_CLOTH_PROXY, BI_E_REAL_CLOTH: material-aware merge=forbidden; fidelities=REAL_CLOTH_ASSET_REQUIRED_NOT_PRESENT, UNCALIBRATED_SURFACE_DEFORMABLE_PROXY
- BI_E_OPEN_MESH_PROXY, BI_E_REAL_MESH: material-aware merge=forbidden; fidelities=REAL_MESH_ASSET_REQUIRED_NOT_PRESENT, UNCALIBRATED_PERFORATED_SURFACE_NET_PROXY
- BI_E_PVC_OVERLAP_PROXY, BI_E_REAL_PVC: material-aware merge=forbidden; fidelities=REAL_PVC_ASSET_REQUIRED_NOT_PRESENT, UNCALIBRATED_SURFACE_PVC_PROXY

## Remaining frozen V5 budget

- Maximum G1 launches: 1000; already launched: 36; remaining: 964.
- Registered target deficit even at 100% valid yield: 856 valid episodes.
- Registered nominal deficit at 80% yield: 1311 launches.
- 964 remaining launches would yield at most about 771 valid episodes at 80% yield.

## Minimum and balanced options after approval

- Minimum feasibility check: 50 unique proxy physical configurations x 3 seeds = 150 physics checks, then one valid episode per configuration (50 episodes; about 63 nominal launches at 80% yield). This is a coverage smoke benchmark, not a formal ranking.
- A balanced plan within the remaining launch ceiling would target 750 valid episodes (15 per unique proxy configuration), requiring about 938 nominal launches at 80% yield. It still cannot satisfy the registered full target and is not an authorization.

No Cartesian-product expansion was launched. A future approved design should use the 50 unique proxy configurations as strata, a documented partial-factorial/LHS selection for parameter combinations, and separate confirmation configurations.
