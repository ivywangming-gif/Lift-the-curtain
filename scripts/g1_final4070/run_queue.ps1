param(
    [string]$ConfigPath = 'configs\g1_final4070\physical_matrix.json',
    [string[]]$Batch = @(),
    [int]$MaxNewLaunches = 48
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$config = Get-Content -LiteralPath (Join-Path $workspace $ConfigPath) -Raw | ConvertFrom-Json
$trialsRoot = Join-Path $workspace 'outputs\g1_final4070\trials'
$queueStatePath = Join-Path $workspace 'outputs\g1_final4070\queue_state.json'
$wrapper = Join-Path $workspace 'scripts\g1_final4070\run_physical_trial.ps1'
$launched = 0
$records = @()

foreach ($trial in ($config.trials | Sort-Object order)) {
    if ($Batch.Count -gt 0 -and $trial.batch -notin $Batch) { continue }
    $runRoot = Join-Path $trialsRoot $trial.trial_id
    if (Test-Path -LiteralPath (Join-Path $runRoot 'summary.json')) {
        $records += [ordered]@{ trial_id=$trial.trial_id; status='SKIPPED_COMPLETE'; time=(Get-Date).ToString('o') }
        continue
    }
    if ((Test-Path -LiteralPath $runRoot) -and (Get-ChildItem -LiteralPath $runRoot -Force -ErrorAction SilentlyContinue)) {
        $records += [ordered]@{ trial_id=$trial.trial_id; status='SKIPPED_NONEMPTY_INCOMPLETE_REQUIRES_AUDIT'; time=(Get-Date).ToString('o') }
        continue
    }
    if ($launched -ge $MaxNewLaunches) { break }
    $record = [ordered]@{ trial_id=$trial.trial_id; status='STARTED'; started_at=(Get-Date).ToString('o') }
    $records += $record
    [ordered]@{ schema='lift-the-curtain.g1-final4070-queue-state.v1'; updated_at=(Get-Date).ToString('o'); records=$records } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $queueStatePath -Encoding utf8
    & powershell -NoProfile -ExecutionPolicy Bypass -File $wrapper -TrialId $trial.trial_id -Scenario $trial.scenario -Policy $trial.policy -ProbeSide $trial.probe_side -Seed $trial.seed -Batch $trial.batch -ConditionId $trial.condition_id -EvidenceLevel 'MATCHED_INITIAL_CONDITION' -OverallTimeoutSeconds 900
    $record.status = if ($LASTEXITCODE -eq 0) { 'PROCESS_COMPLETE' } else { 'PROCESS_NONZERO' }
    $record.finished_at = (Get-Date).ToString('o')
    $record.exit_code = $LASTEXITCODE
    $launched += 1
    [ordered]@{ schema='lift-the-curtain.g1-final4070-queue-state.v1'; updated_at=(Get-Date).ToString('o'); records=$records } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $queueStatePath -Encoding utf8
    & 'C:\conda_envs\isaaclab\python.exe' (Join-Path $workspace 'scripts\g1_final4070\analyze_campaign.py')
}

