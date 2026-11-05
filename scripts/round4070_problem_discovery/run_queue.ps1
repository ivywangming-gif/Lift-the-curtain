param(
    [ValidateSet('all','baseline','exploration')] [string]$Stage = 'all',
    [int]$MaxTrials = 48,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$preregPath = Join-Path $workspace 'configs\round4070_problem_discovery\PREREGISTRATION.json'
$prereg = Get-Content -LiteralPath $preregPath -Raw | ConvertFrom-Json
$selected = @($prereg.trials | Where-Object { $Stage -eq 'all' -or $_.stage -eq $Stage } | Select-Object -First $MaxTrials)
$launched = 0; $skipped = 0; $failed = 0
foreach ($trial in $selected) {
    $summary = Join-Path $workspace ("outputs\round4070_problem_discovery\trials\{0}\summary.json" -f $trial.trial_id)
    if (Test-Path -LiteralPath $summary) {
        Write-Host ("SKIP_COMPLETE {0}" -f $trial.trial_id)
        $skipped += 1
        continue
    }
    $args = @{
        TrialId=[string]$trial.trial_id; Scenario=[string]$trial.scenario; Policy=[string]$trial.policy
        ProbeSide=[string]$trial.probe_side; Stage=[string]$trial.stage; ConditionId=[string]$trial.condition_id
        Seed=[int]$trial.seed; ResetRootY=[double]$trial.reset_root_y; ResetRootYaw=[double]$trial.reset_root_yaw
    }
    if ($DryRun) { $args.DryRun = $true }
    & (Join-Path $PSScriptRoot 'run_trial.ps1') @args
    $exitCode = $LASTEXITCODE
    $launched += 1
    if ($exitCode -ne 0) { $failed += 1 }
    [ordered]@{
        updated_at=(Get-Date).ToString('o'); selected_count=$selected.Count; launched=$launched
        skipped_complete=$skipped; nonzero_exit_count=$failed; last_trial=[string]$trial.trial_id
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $workspace 'outputs\round4070_problem_discovery\queue_state.json') -Encoding utf8
}
Write-Host ("QUEUE_DONE selected={0} launched={1} skipped={2} nonzero={3}" -f $selected.Count,$launched,$skipped,$failed)
if ($failed -gt 0) { exit 2 }
