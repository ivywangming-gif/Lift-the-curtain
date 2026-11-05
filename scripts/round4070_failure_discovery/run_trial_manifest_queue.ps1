param(
    [ValidateSet('baseline', 'discovery', 'confirmation', 'reserved', 'all')]
    [string]$Stage = 'discovery',
    [string[]]$TrialId = @(),
    [string[]]$ConditionId = @(),
    [int]$MaxTrials = 48,
    [int]$OverallTimeoutSeconds = 1200,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$manifestPath = Join-Path $workspace 'configs\round4070_failure_discovery\PREREGISTERED_MANIFEST.json'
$physicsResultsPath = Join-Path $workspace 'research\round4070_failure_discovery\PHYSICS_GATE_RESULTS.csv'
$runner = Join-Path $PSScriptRoot 'run_trial.ps1'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $physicsResultsPath)) { throw "Physics results are required before G1 launch: $physicsResultsPath" }
$physics = @{}
foreach ($row in @(Import-Csv -LiteralPath $physicsResultsPath)) { $physics[[string]$row.gate_id] = $row }

$selected = @($manifest.trials | Where-Object {
    ($Stage -eq 'all' -or [string]$_.stage -eq $Stage) -and
    ((@($TrialId).Count -eq 0) -or (@($TrialId) -contains [string]$_.trial_id)) -and
    ((@($ConditionId).Count -eq 0) -or (@($ConditionId) -contains [string]$_.condition_id))
})
if ($selected.Count -eq 0) { throw 'No manifest trials match the requested filter.' }

$launched = 0
$invalidByCondition = @{}
$results = @()
foreach ($trial in $selected) {
    $trialRoot = Join-Path $workspace ("outputs\round4070_failure_discovery\trials\{0}" -f $trial.trial_id)
    if (Test-Path -LiteralPath (Join-Path $trialRoot 'summary.json')) {
        $results += [pscustomobject]@{ trial_id=$trial.trial_id; status='SKIP_COMPLETE'; exit_code=0; gate_status=$null }
        continue
    }
    $gate = $physics[[string]$trial.gate_id]
    if ($null -eq $gate -or [string]$gate.gate_status -ne 'PASS_PHYSICS_GATE') {
        $gateStatus = if ($null -eq $gate) { 'MISSING' } else { [string]$gate.gate_status }
        $results += [pscustomobject]@{ trial_id=$trial.trial_id; status='SKIP_GATE_NOT_PASS'; exit_code=$null; gate_status=$gateStatus }
        continue
    }
    if ($launched -ge $MaxTrials) { throw "Queue-local launch limit reached: $MaxTrials" }

    $params = @{
        TrialId=[string]$trial.trial_id; Scenario=[string]$trial.scenario; ProbeSide=[string]$trial.probe_side
        Stage=[string]$trial.stage; Group=[string]$trial.group; ConditionId=[string]$trial.condition_id
        Seed=[int]$trial.seed; ResetRootY=[double]$trial.reset_root_y; ResetRootYaw=[double]$trial.reset_root_yaw
        OverallTimeoutSeconds=$OverallTimeoutSeconds
    }
    foreach ($mapping in @(
        @{manifest='hidden_rigid_x'; parameter='HiddenRigidX'}, @{manifest='hidden_rigid_y'; parameter='HiddenRigidY'},
        @{manifest='hidden_rigid_width'; parameter='HiddenRigidWidth'}, @{manifest='hidden_rigid_height'; parameter='HiddenRigidHeight'},
        @{manifest='hidden_rigid_thickness'; parameter='HiddenRigidThickness'}, @{manifest='hidden_rigid_bottom_z'; parameter='HiddenRigidBottomZ'},
        @{manifest='curtain_length'; parameter='CurtainLength'}, @{manifest='curtain_top_z'; parameter='CurtainTopZ'},
        @{manifest='density_scale_multiplier'; parameter='DensityScaleMultiplier'}, @{manifest='curtain_damping'; parameter='CurtainDamping'},
        @{manifest='curtain_bend_stiffness'; parameter='CurtainBendStiffness'}, @{manifest='curtain_stretch_stiffness'; parameter='CurtainStretchStiffness'},
        @{manifest='curtain_shear_stiffness'; parameter='CurtainShearStiffness'}
    )) {
        $value = $trial.($mapping.manifest)
        if ($null -ne $value -and [string]$value -ne '') { $params[$mapping.parameter] = [double]$value }
    }
    if ($DryRun) { $params.DryRun = $true }
    & $runner @params
    $exitCode = $LASTEXITCODE
    $launched += 1
    $summaryPath = Join-Path $trialRoot 'summary.json'
    $metadataPath = Join-Path $trialRoot 'trial_metadata.json'
    $status = if (Test-Path -LiteralPath $summaryPath) { 'RECORDED' } elseif ($DryRun) { 'DRY_RUN' } else { 'RUNTIME_INVALID' }
    $results += [pscustomobject]@{ trial_id=$trial.trial_id; status=$status; exit_code=$exitCode; gate_status=[string]$gate.gate_status }
    if (-not $DryRun -and $status -eq 'RUNTIME_INVALID') {
        $key = [string]$trial.condition_id
        if (-not $invalidByCondition.ContainsKey($key)) { $invalidByCondition[$key] = 0 }
        $invalidByCondition[$key] += 1
        if ($invalidByCondition[$key] -ge 2) {
            Write-Warning "Stopping condition after two runtime-invalid launches: $key"
            break
        }
    }
}

$statePath = Join-Path $workspace 'outputs\round4070_failure_discovery\trial_queue_state.json'
[ordered]@{
    updated_at=(Get-Date).ToString('o'); stage=$Stage; trial_filter=$TrialId; condition_filter=$ConditionId
    launched_in_queue=$launched; results=$results
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $statePath -Encoding utf8
Write-Host ($results | ConvertTo-Csv -NoTypeInformation | Out-String)
