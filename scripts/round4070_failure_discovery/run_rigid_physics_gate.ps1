param(
    [int]$Seed = 61,
    [int]$Steps = 720,
    [int]$OverallTimeoutSeconds = 900,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace 'outputs\round4070_failure_discovery\physics'))
$gate = 'RIGID_CONTROL'
$runner = Join-Path $PSScriptRoot 'validate_rigid_panel.py'
$guard = Join-Path $PSScriptRoot 'run_guarded_with_resources.ps1'

function Invoke-RigidMode([string]$Mode) {
    $runRoot = [System.IO.Path]::GetFullPath((Join-Path $outputRoot (Join-Path $gate $Mode)))
    if (-not $runRoot.StartsWith($outputRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Rigid gate path escapes output root: $runRoot" }
    if (Test-Path -LiteralPath (Join-Path $runRoot 'summary.json')) { Write-Host "SKIP_COMPLETE RIGID_CONTROL $Mode"; return 0 }
    if (Test-Path -LiteralPath $runRoot) {
        $existing = @(Get-ChildItem -LiteralPath $runRoot -Force -ErrorAction SilentlyContinue)
        $placeholderOnly = ($existing.Count -eq 1 -and $existing[0].Name -eq 'command.json')
        if ($existing.Count -gt 0 -and -not $placeholderOnly) { throw "Rigid gate directory is non-empty but incomplete: $runRoot" }
    }
    New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
    $args = @('--mode', $Mode, '--run-root', $runRoot, '--condition-id', 'RIGID_CONTROL', '--seed', [string]$Seed, '--steps', [string]$Steps, '--headless')
    [ordered]@{schema='lift-the-curtain.round4070-failure-discovery-rigid-physics-command.v1'; gate_id=$gate; mode=$Mode; seed=$Seed; run_root=$runRoot; args=$args; dry_run=[bool]$DryRun} |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'command.json') -Encoding utf8
    if ($DryRun) { return 0 }
    & $guard -ScriptPath $runner -RunRoot $runRoot -ScriptArgs $args -OverallTimeoutSeconds $OverallTimeoutSeconds | Out-Host
    return $LASTEXITCODE
}

$noExit = Invoke-RigidMode 'no_probe'
$noSummaryPath = Join-Path $outputRoot 'RIGID_CONTROL\no_probe\summary.json'
$noStatus = if (Test-Path -LiteralPath $noSummaryPath) { (Get-Content -LiteralPath $noSummaryPath -Raw | ConvertFrom-Json).status } else { 'MISSING' }
$probeExit = $null
$probeStatus = 'SKIPPED_NO_PROBE_GATE'
if (-not $DryRun -and $noExit -eq 0 -and $noStatus -eq 'PASS_PHYSICS_SANITY') {
    $probeExit = Invoke-RigidMode 'probe'
    $probeSummaryPath = Join-Path $outputRoot 'RIGID_CONTROL\probe\summary.json'
    if (Test-Path -LiteralPath $probeSummaryPath) { $probeStatus = (Get-Content -LiteralPath $probeSummaryPath -Raw | ConvertFrom-Json).status }
} elseif ($DryRun) {
    $probeStatus = 'DRY_RUN'
}
$result = [pscustomobject]@{ gate_id=$gate; no_probe_exit=$noExit; no_probe_status=$noStatus; probe_exit=$probeExit; probe_status=$probeStatus }
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputRoot 'RIGID_CONTROL\queue_result.json') -Encoding utf8
$result | ConvertTo-Json -Depth 8
if ($noStatus -ne 'PASS_PHYSICS_SANITY' -or ($probeStatus -notin @('PASS_PHYSICS_SANITY', 'DRY_RUN', 'SKIPPED_NO_PROBE_GATE'))) { exit 2 }
