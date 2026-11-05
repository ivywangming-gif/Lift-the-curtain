param(
    [Parameter(Mandatory = $true)] [string]$GateId,
    [Parameter(Mandatory = $true)] [string]$Scenario,
    [Parameter(Mandatory = $true)] [ValidateSet('no_probe', 'probe')] [string]$Mode,
    [int]$Seed = 61,
    [string]$ConditionId = 'UNSPECIFIED',
    [double]$HiddenRigidX = [double]::NaN,
    [double]$HiddenRigidY = [double]::NaN,
    [double]$HiddenRigidWidth = [double]::NaN,
    [double]$HiddenRigidHeight = [double]::NaN,
    [double]$HiddenRigidThickness = [double]::NaN,
    [double]$HiddenRigidBottomZ = [double]::NaN,
    [double]$CurtainLength = 2.00,
    [double]$CurtainTopZ = [double]::NaN,
    [double]$DensityScaleMultiplier = 1.0,
    [double]$CurtainDamping = [double]::NaN,
    [double]$CurtainBendStiffness = [double]::NaN,
    [double]$CurtainStretchStiffness = [double]::NaN,
    [double]$CurtainShearStiffness = [double]::NaN,
    [ValidateSet('A0_TOP_EDGE_ONLY','A1_TOP_EDGE_BOTTOM_TWO_CORNERS','A2_FOUR_CORNERS_ONLY','A3_FULL_PERIMETER_FIXED')] [string]$StructuralTopology = 'A0_TOP_EDGE_ONLY',
    [double]$SurfaceThickness = 0.002,
    [int]$Steps = 720,
    [int]$OverallTimeoutSeconds = 900,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace 'outputs\round4070_fixed10cm_judgment\physics'))
$runRoot = [System.IO.Path]::GetFullPath((Join-Path $outputRoot (Join-Path $GateId $Mode)))
if (-not $runRoot.StartsWith($outputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Physics gate path escapes campaign output root: $runRoot"
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'summary.json')) {
    Write-Host "SKIP_COMPLETE $GateId $Mode"
    exit 0
}
if (Test-Path -LiteralPath $runRoot) {
    $existing = @(Get-ChildItem -LiteralPath $runRoot -Force -ErrorAction SilentlyContinue)
    $dryRunPlaceholderOnly = ($existing.Count -eq 1 -and $existing[0].Name -eq 'command.json')
    if ($existing.Count -gt 0 -and -not $dryRunPlaceholderOnly) {
        throw "Physics gate directory is non-empty but incomplete: $runRoot"
    }
}
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$script = Join-Path $workspace 'scripts\round4070_fixed10cm_judgment\validate_single_panel.py'
$guard = Join-Path $workspace 'scripts\round4070_fixed10cm_judgment\run_guarded_with_resources.ps1'
$args = @(
    '--scenario', $Scenario,
    '--mode', $Mode,
    '--run-root', $runRoot,
    '--seed', [string]$Seed,
    '--condition-id', $ConditionId,
    '--curtain-length', [string]$CurtainLength,
    '--density-scale-multiplier', [string]$DensityScaleMultiplier,
    '--structural-topology', $StructuralTopology,
    '--surface-thickness', [string]$SurfaceThickness,
    '--steps', [string]$Steps,
    '--headless'
)
if (-not [double]::IsNaN($HiddenRigidX)) { $args += @('--hidden-rigid-x', [string]$HiddenRigidX) }
if (-not [double]::IsNaN($HiddenRigidY)) { $args += @('--hidden-rigid-y', [string]$HiddenRigidY) }
if (-not [double]::IsNaN($HiddenRigidWidth)) { $args += @('--hidden-rigid-width', [string]$HiddenRigidWidth) }
if (-not [double]::IsNaN($HiddenRigidHeight)) { $args += @('--hidden-rigid-height', [string]$HiddenRigidHeight) }
if (-not [double]::IsNaN($HiddenRigidThickness)) { $args += @('--hidden-rigid-thickness', [string]$HiddenRigidThickness) }
if (-not [double]::IsNaN($HiddenRigidBottomZ)) { $args += @('--hidden-rigid-bottom-z', [string]$HiddenRigidBottomZ) }
if (-not [double]::IsNaN($CurtainTopZ)) { $args += @('--curtain-top-z', [string]$CurtainTopZ) }
if (-not [double]::IsNaN($CurtainDamping)) { $args += @('--curtain-damping', [string]$CurtainDamping) }
if (-not [double]::IsNaN($CurtainBendStiffness)) { $args += @('--curtain-bend-stiffness', [string]$CurtainBendStiffness) }
if (-not [double]::IsNaN($CurtainStretchStiffness)) { $args += @('--curtain-stretch-stiffness', [string]$CurtainStretchStiffness) }
if (-not [double]::IsNaN($CurtainShearStiffness)) { $args += @('--curtain-shear-stiffness', [string]$CurtainShearStiffness) }
if ($Mode -eq 'probe') {
    $control = Join-Path $outputRoot (Join-Path $GateId 'no_probe\summary.json')
    if (-not (Test-Path -LiteralPath $control)) { throw "Matched no-probe summary is required first: $control" }
    $args += @('--matched-no-probe-summary', $control)
}

[ordered]@{
    schema = 'lift-the-curtain.round4070-fixed10cm-physics-command.v1'
    gate_id = $GateId; condition_id = $ConditionId; scenario = $Scenario; mode = $Mode; seed = $Seed
    script = $script; guard = $guard; run_root = $runRoot; args = $args; dry_run = [bool]$DryRun
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'command.json') -Encoding utf8
if ($DryRun) { $args -join ' '; exit 0 }

& $guard -ScriptPath $script -RunRoot $runRoot -ScriptArgs $args -OverallTimeoutSeconds $OverallTimeoutSeconds
exit $LASTEXITCODE
