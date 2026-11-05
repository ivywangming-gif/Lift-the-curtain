param(
    [Parameter(Mandatory = $true)] [string]$TrialId,
    [Parameter(Mandatory = $true)] [string]$Scenario,
    [Parameter(Mandatory = $true)] [ValidateSet('left', 'right')] [string]$ProbeSide,
    [string]$Stage = 'UNSPECIFIED',
    [string]$Group = 'UNSPECIFIED',
    [string]$ConditionId = 'UNSPECIFIED',
    [int]$Seed = 61,
    [double]$ResetRootY = 0.0,
    [double]$ResetRootYaw = 0.0,
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
    [int]$OverallTimeoutSeconds = 1200,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace 'outputs\round4070_failure_discovery\trials'))
$runRoot = [System.IO.Path]::GetFullPath((Join-Path $outputRoot $TrialId))
if (-not $runRoot.StartsWith($outputRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Trial path escapes campaign output root: $runRoot" }
$manifestPath = Join-Path $workspace 'configs\round4070_failure_discovery\PREREGISTERED_MANIFEST.json'
if (Test-Path -LiteralPath $manifestPath) {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $plannedIds = @($manifest.trials | ForEach-Object { [string]$_.trial_id })
    if ($plannedIds -notcontains $TrialId) { throw "Trial is not in the frozen manifest: $TrialId" }
}
$ledgerPath = Join-Path $workspace 'outputs\round4070_failure_discovery\launch_ledger.json'
$hardLimit = 48
if (Test-Path -LiteralPath $ledgerPath) {
    $ledger = Get-Content -LiteralPath $ledgerPath -Raw | ConvertFrom-Json
    if ([int]$ledger.launched_count -ge $hardLimit -and -not (Test-Path -LiteralPath (Join-Path $runRoot 'summary.json'))) {
        throw "New G1 hard budget reached: $hardLimit"
    }
} else {
    $ledger = [pscustomobject]@{ schema='lift-the-curtain.round4070-failure-discovery-launch-ledger.v1'; launched_count=0; trial_ids=@(); updated_at=(Get-Date).ToString('o') }
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'summary.json')) { Write-Host "SKIP_COMPLETE $TrialId"; exit 0 }
if (Test-Path -LiteralPath $runRoot) {
    $existing = @(Get-ChildItem -LiteralPath $runRoot -Force -ErrorAction SilentlyContinue)
    if ($existing.Count -gt 0) { throw "Trial directory is non-empty but incomplete; preserve it: $runRoot" }
}
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$metadata = [ordered]@{
    schema='lift-the-curtain.round4070-failure-discovery-trial.v1'; trial_id=$TrialId; stage=$Stage; group=$Group; condition_id=$ConditionId
    scenario_for_scene_builder=$Scenario; probe_side=$ProbeSide; seed=$Seed; reset_root_y=$ResetRootY; reset_root_yaw=$ResetRootYaw
    started_at=(Get-Date).ToString('o'); historical_outputs_modified=$false; deployment_decision_must_not_use_scene_identity=$true
    formal_physics_settings_changed=$false; frozen_runner='scripts/round4070_failure_discovery/run_episode.py'
    overrides=[ordered]@{hidden_rigid_x=$HiddenRigidX; hidden_rigid_y=$HiddenRigidY; hidden_rigid_width=$HiddenRigidWidth; hidden_rigid_height=$HiddenRigidHeight; hidden_rigid_thickness=$HiddenRigidThickness; hidden_rigid_bottom_z=$HiddenRigidBottomZ; curtain_length=$CurtainLength; curtain_top_z=$CurtainTopZ; density_scale_multiplier=$DensityScaleMultiplier; curtain_damping=$CurtainDamping; curtain_bend_stiffness=$CurtainBendStiffness; curtain_stretch_stiffness=$CurtainStretchStiffness; curtain_shear_stiffness=$CurtainShearStiffness}
}
$metadata | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $runRoot 'trial_metadata.json') -Encoding utf8

$script = Join-Path $workspace 'scripts\round4070_failure_discovery\run_episode.py'
$guard = Join-Path $workspace 'scripts\round4070_failure_discovery\run_guarded_with_resources.ps1'
$args = @(
    '--run-root', $runRoot, '--seed', [string]$Seed, '--probe-side', $ProbeSide,
    '--high-level-policy', 'r1', '--reset-root-y', [string]$ResetRootY, '--reset-root-yaw', [string]$ResetRootYaw,
    '--precontact-offset', '0.40', '--precontact-region-x-min', '0.36', '--precontact-region-x-max', '0.48',
    '--precontact-region-y-half-width', '0.10', '--precontact-region-speed-max', '0.06',
    '--round15-progress-monitor', '--research-feasibility-post-clearance-replan', '--condition-id', $ConditionId,
    '--curtain-length', [string]$CurtainLength, '--density-scale-multiplier', [string]$DensityScaleMultiplier,
    '--headless', '--scenario', $Scenario
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

[ordered]@{schema='lift-the-curtain.round4070-failure-discovery-command.v1'; script=$script; guard=$guard; run_root=$runRoot; args=$args; dry_run=[bool]$DryRun} |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'command.json') -Encoding utf8
if ($DryRun) { $args -join ' '; exit 0 }

$trialIds = @($ledger.trial_ids | ForEach-Object { [string]$_ })
if ($trialIds -notcontains $TrialId) { $trialIds += $TrialId; $ledger.launched_count = $trialIds.Count; $ledger.trial_ids = $trialIds; $ledger.updated_at = (Get-Date).ToString('o'); $ledger | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ledgerPath -Encoding utf8 }
& $guard -ScriptPath $script -RunRoot $runRoot -ScriptArgs $args -OverallTimeoutSeconds $OverallTimeoutSeconds
$exitCode = $LASTEXITCODE
$metadata.finished_at=(Get-Date).ToString('o'); $metadata.exit_code=$exitCode; $metadata.status=if($exitCode -eq 0){'PROCESS_COMPLETE'}elseif(Test-Path (Join-Path $runRoot 'summary.json')){'EPISODE_RECORDED'}else{'RUNTIME_INVALID'}
$metadata | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $runRoot 'trial_metadata.json') -Encoding utf8
exit $exitCode
