param(
    [int]$Seed = 4070,
    [int]$Steps = 720,
    [int]$OverallTimeoutSeconds = 900,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace 'outputs\round4070_fixed_probe_benchmark\p1'))
$script = Join-Path $PSScriptRoot 'validate_p1_flexible_scene.py'
$guard = Join-Path $PSScriptRoot 'run_guarded_with_resources.ps1'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$scenes = @(
    [pscustomobject]@{ name='SPLIT_CENTER_CURTAIN'; family='PVC_INDEPENDENT_SHEETS_PROXY'; material_contract='surface-deformable proxy, not real PVC' },
    [pscustomobject]@{ name='LEFT_OPEN_RIGHT_CONSTRAINED'; family='PVC_INDEPENDENT_SHEETS_PROXY'; material_contract='surface-deformable proxy, not real PVC' },
    [pscustomobject]@{ name='RIGHT_OPEN_LEFT_CONSTRAINED'; family='PVC_INDEPENDENT_SHEETS_PROXY'; material_contract='surface-deformable proxy, not real PVC' },
    [pscustomobject]@{ name='PVC_INDEPENDENT_STRIPS'; family='ARTICULATED_STRIP_SURROGATE'; material_contract='articulated strips, not real PVC' }
)
$results = @()
foreach ($scene in $scenes) {
    $sceneRoot = Join-Path $outputRoot $scene.name
    $noRoot = Join-Path $sceneRoot 'no_probe'
    $probeRoot = Join-Path $sceneRoot 'probe'
    foreach ($mode in @('no_probe', 'probe')) {
        $runRoot = if ($mode -eq 'no_probe') { $noRoot } else { $probeRoot }
        if (Test-Path -LiteralPath (Join-Path $runRoot 'summary.json')) { continue }
        New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
        $args = @('--scene', $scene.name, '--mode', $mode, '--run-root', $runRoot, '--seed', [string]$Seed, '--steps', [string]$Steps, '--headless')
        if ($mode -eq 'probe') { $args += @('--matched-no-probe-summary', (Join-Path $noRoot 'summary.json')) }
        [ordered]@{
            schema='lift-the-curtain.round4070-fixed-probe-p1-command.v1'; scene=$scene.name; mode=$mode
            model_contract=$scene.family; material_contract=$scene.material_contract
            script=$script; guard=$guard; run_root=$runRoot; args=$args; dry_run=[bool]$DryRun
        } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'command.json') -Encoding utf8
        if (-not $DryRun) {
            & $guard -ScriptPath $script -RunRoot $runRoot -ScriptArgs $args -OverallTimeoutSeconds $OverallTimeoutSeconds
            $exitCode = $LASTEXITCODE
        } else { $exitCode = 0 }
        $summaryPath = Join-Path $runRoot 'summary.json'
        $summary = if (Test-Path -LiteralPath $summaryPath) { Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json } else { $null }
        $results += [pscustomobject]@{
            scene=$scene.name; mode=$mode; exit_code=$exitCode
            status=if ($summary) { [string]$summary.status } elseif ($DryRun) { 'DRY_RUN' } else { 'MISSING' }
            model_contract=$scene.family; material_contract=$scene.material_contract; run_root=$runRoot
        }
    }
}
$report = [ordered]@{
    schema='lift-the-curtain.round4070-fixed-probe-p1-gate-results.v1'
    campaign='Round 4070-E'; updated_at=(Get-Date).ToString('o')
    p1_is_physics_only=$true; p1_failures_cannot_invalidate_p0=$true
    real_pvc_modeled=$false; results=$results
}
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $outputRoot 'P1_GATE_RESULTS.json') -Encoding utf8
$results | ConvertTo-Csv -NoTypeInformation | Write-Host
if (-not $DryRun -and @($results | Where-Object { $_.mode -eq 'no_probe' -and $_.status -ne 'PASS_PHYSICS_SANITY' }).Count -gt 0) { exit 2 }
