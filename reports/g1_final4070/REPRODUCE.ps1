param(
    [switch]$RunPhysical
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location -LiteralPath $workspace
$python = 'C:\conda_envs\isaaclab\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Isaac Lab Python not found: $python"
}

if ($RunPhysical) {
    # The queue skips completed trial directories and remains serial.
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/g1_final4070/run_queue.ps1 -Batch RESOURCE_BASELINE -MaxNewLaunches 48
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/g1_final4070/run_queue.ps1 -Batch MATCHED_ACTION -MaxNewLaunches 48
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/g1_final4070/run_queue.ps1 -Batch MATCHED_STATE -MaxNewLaunches 48
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/g1_final4070/run_queue.ps1 -Batch KEY_REPEAT -MaxNewLaunches 48
}

& $python scripts/g1_final4070/analyze_campaign.py --repo $workspace
& $python scripts/g1_final4070/finalize_campaign.py
& $python -m pytest -q tests/g1_interactive_nav_baseline --disable-warnings --maxfail=1
& $python scripts/test_round27_contracts.py

Write-Output 'Final RTX 4070 campaign analysis complete.'

