param(
    [Parameter(Mandatory = $true)] [string]$TrialId,
    [string]$Stage = 'UNSPECIFIED',
    [string]$Group = 'UNSPECIFIED',
    [string]$ConditionId = 'UNSPECIFIED',
    [string]$Scenario = 'nominal',
    [ValidateSet('left', 'right')] [string]$ProbeSide = 'right',
    [int]$Seed = 4070,
    [double]$ResetRootY = 0.0,
    [double]$ResetRootYaw = 0.0,
    [double]$HiddenRigidX = [double]::NaN,
    [double]$DensityScaleMultiplier = 1.0,
    [ValidateSet('PROBE_10CM')] [string]$ProbeDepthLevel = 'PROBE_10CM',
    [double]$Probe10cmJointStepRad = 0.85,
    [ValidateSet('A0_TOP_EDGE_ONLY','A1_TOP_EDGE_BOTTOM_TWO_CORNERS','A2_FOUR_CORNERS_ONLY','A3_FULL_PERIMETER_FIXED')] [string]$StructuralTopology = 'A0_TOP_EDGE_ONLY',
    [double]$SurfaceThickness = 0.002,
    [int]$OverallTimeoutSeconds = 900,
    [switch]$DiagnosticNoObstacle,
    [switch]$ProbeValidationOnly,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace 'outputs\round4070_fixed_probe_benchmark\trials'))
$runRoot = [System.IO.Path]::GetFullPath((Join-Path $outputRoot $TrialId))
if (-not $runRoot.StartsWith($outputRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Trial path escapes E output root: $runRoot" }
$manifestPath = Join-Path $workspace 'configs\round4070_fixed_probe_benchmark\PREREGISTERED_MANIFEST.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "E manifest is missing: $manifestPath" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$trialPlan = @($manifest.trials | Where-Object { [string]$_.trial_id -eq $TrialId }) | Select-Object -First 1
if ($null -eq $trialPlan) { throw "Trial is not in the frozen Round 4070-E manifest: $TrialId" }
# The frozen manifest is authoritative. Resolve all scene and probe parameters
# from it so a direct TrialId invocation cannot silently use parameter defaults.
$Stage = [string]$trialPlan.phase
$Group = [string]$trialPlan.group
$ConditionId = [string]$trialPlan.condition_id
$Scenario = [string]$trialPlan.scenario
$ProbeSide = [string]$trialPlan.probe_side
$Seed = [int]$trialPlan.seed
$ResetRootY = [double]$trialPlan.reset_root_y
$ResetRootYaw = [double]$trialPlan.reset_root_yaw
$ProbeDepthLevel = [string]$trialPlan.probe_depth_level
$Probe10cmJointStepRad = [double]$trialPlan.probe_10cm_joint_step_rad
$StructuralTopology = [string]$trialPlan.structural_topology
$DiagnosticNoObstacle = [bool]$trialPlan.diagnostic_no_obstacle
if ($trialPlan.PSObject.Properties.Name -contains 'density_scale_multiplier' -and $null -ne $trialPlan.density_scale_multiplier) {
    $DensityScaleMultiplier = [double]$trialPlan.density_scale_multiplier
}
if ($trialPlan.PSObject.Properties.Name -contains 'hidden_rigid_x' -and $null -ne $trialPlan.hidden_rigid_x) {
    $HiddenRigidX = [double]$trialPlan.hidden_rigid_x
} else {
    $HiddenRigidX = [double]::NaN
}
$gatePath = Join-Path $workspace 'configs\round4070_fixed_probe_benchmark\PHYSICS_GATE_STATUS.json'
$gateStatus = if (Test-Path -LiteralPath $gatePath) { Get-Content -LiteralPath $gatePath -Raw | ConvertFrom-Json } else { $null }
$registeredGate = [string]$trialPlan.physics_gate_id
$resolvedGateStatus = if ($gateStatus -and $gateStatus.gates.PSObject.Properties.Name -contains $registeredGate) { [string]$gateStatus.gates.$registeredGate.status } else { 'PENDING' }
$isFormalCore = [bool]$trialPlan.formal_core
if ($isFormalCore -and $resolvedGateStatus -notin @('PASS', 'PASS_E_PHYSICS', 'PASS_INHERITED', 'PASS_INHERITED_C')) {
    throw "E physics gate $registeredGate is not PASS for ${TrialId}: $resolvedGateStatus"
}
$ledgerPath = Join-Path $workspace 'outputs\round4070_fixed_probe_benchmark\launch_ledger.json'
$hardLimit = 200
if (Test-Path -LiteralPath $ledgerPath) {
    $ledger = Get-Content -LiteralPath $ledgerPath -Raw | ConvertFrom-Json
} else {
    $ledger = [pscustomobject]@{ schema='lift-the-curtain.round4070-fixed-probe-launch-ledger.v1'; launched_count=0; trial_ids=@(); updated_at=(Get-Date).ToString('o') }
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'summary.json')) { Write-Host "SKIP_COMPLETE $TrialId"; exit 0 }
if ([int]$ledger.launched_count -ge $hardLimit) { throw "E new G1 hard budget reached: $hardLimit" }
if (Test-Path -LiteralPath $runRoot) {
    $existing = @(Get-ChildItem -LiteralPath $runRoot -Force -ErrorAction SilentlyContinue)
    $existingCommand = Join-Path $runRoot 'command.json'
    $commandIsDryRun = $false
    if (Test-Path -LiteralPath $existingCommand) {
        try {
            $existingCommandJson = Get-Content -LiteralPath $existingCommand -Raw | ConvertFrom-Json
            $commandIsDryRun = [bool]$existingCommandJson.dry_run
        } catch {
            $commandIsDryRun = $false
        }
    }
    $allowedPlaceholderNames = @('command.json', 'trial_metadata.json')
    $hasNoSummary = -not (Test-Path -LiteralPath (Join-Path $runRoot 'summary.json'))
    $hasOnlyPlaceholderFiles = (@($existing | Where-Object { $allowedPlaceholderNames -notcontains $_.Name }).Count -eq 0)
    $dryRunPlaceholderOnly = ($existing.Count -gt 0 -and $commandIsDryRun -and $hasNoSummary -and $hasOnlyPlaceholderFiles)
    if ($existing.Count -gt 0 -and -not $dryRunPlaceholderOnly) { throw "Trial directory is non-empty but incomplete; preserve it: $runRoot" }
}
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$phase = [string]$trialPlan.phase
$runValidationOnly = [bool]$ProbeValidationOnly -or $phase -eq 'PHASE_0' -or [string]$trialPlan.condition_family -in @('control', 'matched_no_wall')
$metadata = [ordered]@{
    schema='lift-the-curtain.round4070-fixed-probe-trial.v1'
    trial_id=$TrialId; phase=$phase; stage=$Stage; group=$Group; condition_id=$ConditionId
    scenario_for_scene_builder=$Scenario; probe_side=$ProbeSide; seed=$Seed
    reset_root_y=$ResetRootY; reset_root_yaw=$ResetRootYaw
    started_at=(Get-Date).ToString('o'); historical_outputs_modified=$false
    deployment_decision_must_not_use_scene_identity=$true
    probe_contract='FIXED_ONE_SHOT_NOMINAL_0.10M_ACTUAL_FEEDBACK_RECORDED'
    runner='scripts/round4070_fixed_probe_benchmark/run_episode.py'
    physics_gate_id=$registeredGate; physics_gate_status=$resolvedGateStatus
    probe_validation_only=$runValidationOnly
}
$metadata | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $runRoot 'trial_metadata.json') -Encoding utf8

$script = Join-Path $workspace 'scripts\round4070_fixed_probe_benchmark\run_episode.py'
$guard = Join-Path $workspace 'scripts\round4070_fixed_probe_benchmark\run_guarded_with_resources.ps1'
$args = @(
    '--run-root', $runRoot, '--seed', [string]$Seed, '--probe-side', $ProbeSide,
    '--high-level-policy', 'r1', '--reset-root-y', [string]$ResetRootY, '--reset-root-yaw', [string]$ResetRootYaw,
    '--precontact-offset', '0.40', '--precontact-region-x-min', '0.36', '--precontact-region-x-max', '0.48',
    '--precontact-region-y-half-width', '0.10', '--precontact-region-speed-max', '0.06',
    '--round15-progress-monitor', '--research-feasibility-post-clearance-replan', '--condition-id', $ConditionId,
    '--curtain-length', '2.00', '--density-scale-multiplier', [string]$DensityScaleMultiplier,
    '--probe-depth-level', $ProbeDepthLevel, '--probe-10cm-joint-step-rad', [string]$Probe10cmJointStepRad,
    '--structural-topology', $StructuralTopology, '--surface-thickness', [string]$SurfaceThickness,
    '--headless', '--scenario', $Scenario
)
if ($DiagnosticNoObstacle) { $args += '--diagnostic-no-obstacle' }
if ($runValidationOnly) { $args += '--probe-validation-only' }
if (-not [double]::IsNaN($HiddenRigidX)) { $args += @('--hidden-rigid-x', [string]$HiddenRigidX) }

[ordered]@{
    schema='lift-the-curtain.round4070-fixed-probe-command.v1'; script=$script; guard=$guard
    run_root=$runRoot; args=$args; dry_run=[bool]$DryRun
    physics_gate_id=$registeredGate; physics_gate_status=$resolvedGateStatus
    formal_core=$isFormalCore; actual_motion_is_not_validity_gate=$true
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'command.json') -Encoding utf8
if ($DryRun) { $args -join ' '; exit 0 }

$trialIds = @($ledger.trial_ids | ForEach-Object { [string]$_ })
if ($trialIds -notcontains $TrialId) {
    $trialIds += $TrialId
    $ledger.launched_count = $trialIds.Count
    $ledger.trial_ids = $trialIds
    $ledger.updated_at = (Get-Date).ToString('o')
    $ledger | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ledgerPath -Encoding utf8
}
& $guard -ScriptPath $script -RunRoot $runRoot -ScriptArgs $args -OverallTimeoutSeconds $OverallTimeoutSeconds
$exitCode = $LASTEXITCODE
$metadata.finished_at=(Get-Date).ToString('o')
$metadata.exit_code=$exitCode
$metadata.status=if($exitCode -eq 0){'PROCESS_COMPLETE'}elseif(Test-Path (Join-Path $runRoot 'summary.json')){'EPISODE_RECORDED'}else{'RUNTIME_INVALID'}
$metadata | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $runRoot 'trial_metadata.json') -Encoding utf8
exit $exitCode
