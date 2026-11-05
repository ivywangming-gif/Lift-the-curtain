param(
    [Parameter(Mandatory = $true)] [string]$TrialId,
    [Parameter(Mandatory = $true)]
    [ValidateSet('nominal','rigid_panel','double_density','bottom_left_anchor','bottom_right_anchor','both_bottom_corners','curtain_hidden_rigid','empty')]
    [string]$Scenario,
    [ValidateSet('r1','r2','d1')] [string]$Policy = 'r1',
    [ValidateSet('left','right')] [string]$ProbeSide = 'right',
    [ValidateSet('EXACT_MATCHED_STATE','MATCHED_INITIAL_CONDITION','RANDOMIZED_GROUP_COMPARISON','RESOURCE_DIAGNOSTIC_ONLY')]
    [string]$EvidenceLevel = 'MATCHED_INITIAL_CONDITION',
    [string]$Batch = 'UNSPECIFIED',
    [string]$ConditionId = 'UNSPECIFIED',
    [int]$Seed = 45,
    [double]$ResetRootY = 0.0,
    [double]$ResetRootYaw = 0.0,
    [int]$OverallTimeoutSeconds = 900,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace 'outputs\g1_final4070\trials'))
$runRoot = Join-Path $outputRoot $TrialId
if (-not $runRoot.StartsWith($outputRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Trial path escapes campaign output root: $runRoot" }
if ((Test-Path -LiteralPath $runRoot) -and -not $DryRun) {
    $existing = Get-ChildItem -LiteralPath $runRoot -Force -ErrorAction SilentlyContinue
    if ($existing) { throw "Trial directory already contains data; choose a new TrialId: $runRoot" }
}
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$metadata = [ordered]@{
    schema='lift-the-curtain.g1-final4070-trial.v1'; trial_id=$TrialId; batch=$Batch; condition_id=$ConditionId
    evidence_level=$EvidenceLevel; scenario_for_scene_builder=$Scenario; policy=$Policy; probe_side=$ProbeSide; seed=$Seed
    reset_root_y=$ResetRootY; reset_root_yaw=$ResetRootYaw; started_at=(Get-Date).ToString('o')
    historical_outputs_modified=$false; deployment_decision_must_not_use_scene_identity=$true; formal_physics_settings_changed=$false
}
$metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'trial_metadata.json') -Encoding utf8
if ($Scenario -eq 'empty') { $scenarioArgs = @('--scenario','nominal','--diagnostic-no-obstacle') } else { $scenarioArgs = @('--scenario',$Scenario) }
$scriptArgs = @(
    '--run-root',$runRoot,'--seed',[string]$Seed,'--probe-side',$ProbeSide,'--high-level-policy',$Policy,
    '--reset-root-y',([string]$ResetRootY),'--reset-root-yaw',([string]$ResetRootYaw),
    '--precontact-offset','0.40','--precontact-region-x-min','0.36','--precontact-region-x-max','0.48',
    '--precontact-region-y-half-width','0.10','--precontact-region-speed-max','0.06',
    '--round15-progress-monitor','--research-feasibility-post-clearance-replan','--headless'
) + $scenarioArgs
$commandRecord = [ordered]@{ script='scripts/test_round27_policy_episode.py'; guard='scripts/g1_final4070/run_guarded_with_resources.ps1'; run_root=$runRoot; args=$scriptArgs; dry_run=[bool]$DryRun }
$commandRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'command.json') -Encoding utf8
if ($DryRun) {
    $metadata.status='DRY_RUN'; $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'trial_metadata.json') -Encoding utf8
    $scriptArgs -join ' '; exit 0
}
$guard = Join-Path $workspace 'scripts\g1_final4070\run_guarded_with_resources.ps1'
$script = Join-Path $workspace 'scripts\test_round27_policy_episode.py'
& $guard -ScriptPath $script -RunRoot $runRoot -ScriptArgs $scriptArgs -OverallTimeoutSeconds $OverallTimeoutSeconds
$exitCode = $LASTEXITCODE
$metadata.finished_at=(Get-Date).ToString('o'); $metadata.exit_code=$exitCode
$metadata.status=if($exitCode -eq 0){'PROCESS_COMPLETE'}elseif(Test-Path (Join-Path $runRoot 'summary.json')){'EPISODE_RECORDED'}else{'RUNTIME_INVALID'}
$metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'trial_metadata.json') -Encoding utf8
exit $exitCode

