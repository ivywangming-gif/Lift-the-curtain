param(
    [Parameter(Mandatory = $true)] [string[]]$TrialId,
    [int]$OverallTimeoutSeconds = 1200,
    [switch]$ProbeValidationOnly
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$manifestDir = Join-Path $workspace 'configs\round4070_fixed_probe_benchmark'
$manifestNames = @('PREREGISTERED_MANIFEST.json')
$plans = @()
foreach ($manifestName in $manifestNames) {
    $candidate = Join-Path $manifestDir $manifestName
    if (Test-Path -LiteralPath $candidate) {
        $plans += @((Get-Content -LiteralPath $candidate -Raw | ConvertFrom-Json).trials)
    }
}
$runner = Join-Path $PSScriptRoot 'run_trial.ps1'
$records = @()

foreach ($wantedId in $TrialId) {
    $trial = @($plans | Where-Object { [string]$_.trial_id -eq [string]$wantedId }) | Select-Object -First 1
    if ($null -eq $trial) { throw "Trial is not in the frozen Round 4070-E manifest: $wantedId" }
    $params = @{
        TrialId = [string]$trial.trial_id
        Scenario = [string]$trial.scenario
        ProbeSide = [string]$trial.probe_side
        Stage = [string]$trial.stage
        Group = [string]$trial.group
        ConditionId = [string]$trial.condition_id
        Seed = [int]$trial.seed
        ResetRootY = [double]$trial.reset_root_y
        ResetRootYaw = [double]$trial.reset_root_yaw
        ProbeDepthLevel = 'PROBE_10CM'
        Probe10cmJointStepRad = [double]$trial.probe_10cm_joint_step_rad
        StructuralTopology = [string]$trial.structural_topology
        OverallTimeoutSeconds = $OverallTimeoutSeconds
        ProbeValidationOnly = $ProbeValidationOnly
        DiagnosticNoObstacle = [bool]$trial.diagnostic_no_obstacle
    }
    if ($trial.PSObject.Properties.Name -contains 'density_scale_multiplier' -and $null -ne $trial.density_scale_multiplier) {
        $params.DensityScaleMultiplier = [double]$trial.density_scale_multiplier
    }
    if ($trial.PSObject.Properties.Name -contains 'hidden_rigid_x' -and $null -ne $trial.hidden_rigid_x) {
        $params.HiddenRigidX = [double]$trial.hidden_rigid_x
    }
    Write-Host ("START_SERIAL {0}" -f $trial.trial_id)
    & $runner @params
    $exitCode = $LASTEXITCODE
    $summaryPath = Join-Path $workspace ("outputs\round4070_fixed_probe_benchmark\trials\{0}\summary.json" -f $trial.trial_id)
    $status = if (Test-Path -LiteralPath $summaryPath) { (Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json).status } else { 'MISSING_SUMMARY' }
    $records += [pscustomobject]@{ trial_id=[string]$trial.trial_id; exit_code=$exitCode; status=$status; summary_path=$summaryPath; finished_at=(Get-Date).ToString('o') }
    Write-Host ("END_SERIAL {0} exit={1} status={2}" -f $trial.trial_id, $exitCode, $status)
}

$outPath = Join-Path $workspace 'outputs\round4070_fixed_probe_benchmark\serial_trial_batch_log.json'
[ordered]@{schema='lift-the-curtain.round4070-fixed-probe-serial-batch.v1'; updated_at=(Get-Date).ToString('o'); records=$records} |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $outPath -Encoding utf8
$records | ConvertTo-Csv -NoTypeInformation | Write-Host
