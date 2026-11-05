param(
    [int]$MaxTrials = 16,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$manifestPath = Join-Path $workspace 'configs\round4070_problem_discovery\CONFIRMATORY_PREREGISTRATION.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$selected = @($manifest.trials | Select-Object -First $MaxTrials)
$launched = 0
$skipped = 0
$failed = 0

foreach ($trial in $selected) {
    $summary = Join-Path $workspace ("outputs\round4070_problem_discovery\trials\{0}\summary.json" -f $trial.trial_id)
    if (Test-Path -LiteralPath $summary) {
        Write-Host ("SKIP_COMPLETE {0}" -f $trial.trial_id)
        $skipped += 1
        continue
    }
    $arguments = @{
        TrialId = [string]$trial.trial_id
        Scenario = [string]$trial.scenario
        Policy = [string]$trial.policy
        ProbeSide = [string]$trial.probe_side
        Stage = [string]$trial.stage
        ConditionId = [string]$trial.condition_id
        Seed = [int]$trial.seed
        ResetRootY = [double]$trial.reset_root_y
        ResetRootYaw = [double]$trial.reset_root_yaw
    }
    if ($DryRun) {
        $arguments.DryRun = $true
    }
    & (Join-Path $PSScriptRoot 'run_trial.ps1') @arguments
    $exitCode = $LASTEXITCODE
    $launched += 1
    if ($exitCode -ne 0) {
        $failed += 1
    }
    [ordered]@{
        updated_at = (Get-Date).ToString('o')
        manifest = $manifestPath
        selected_count = $selected.Count
        launched = $launched
        skipped_complete = $skipped
        nonzero_exit_count = $failed
        last_trial = [string]$trial.trial_id
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $workspace 'outputs\round4070_problem_discovery\confirmatory_queue_state.json') -Encoding utf8
}

Write-Host ("CONFIRMATORY_QUEUE_DONE selected={0} launched={1} skipped={2} nonzero={3}" -f $selected.Count, $launched, $skipped, $failed)
if ($failed -gt 0) {
    exit 2
}
