param([int]$Seed = 47, [switch]$DryRun)
$ErrorActionPreference = 'Stop'
$failures = 0
foreach ($scene in @('split_free_layered24','split_left_limited_layered24','split_right_limited_layered24','strips_3_overlap_layered40')) {
    foreach ($mode in @('no_probe','probe')) {
        $args = @{Scene=$scene;Mode=$mode;Seed=$Seed}
        if ($DryRun) { $args.DryRun = $true }
        & (Join-Path $PSScriptRoot 'run_scene_validation.ps1') @args
        if ($LASTEXITCODE -ne 0) {
            $failures += 1
            if ($mode -eq 'no_probe') { break }
        }
    }
}
if ($failures -gt 0) { exit 2 }
