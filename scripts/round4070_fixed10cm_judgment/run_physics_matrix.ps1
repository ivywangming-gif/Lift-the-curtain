param(
    [int]$Seed = 4070,
    [int]$Steps = 720,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$runner = Join-Path $PSScriptRoot 'run_physics_gate.ps1'
$queueState = Join-Path $workspace 'outputs\round4070_fixed10cm_judgment\physics_queue_state.json'

# P0 gates are deliberately small and one-variable-at-a-time.  A2/A3 are
# genuinely different attachment topologies; they are not aliases for the
# historical bottom-corner scenes.
$conditions = @(
    [pscustomobject]@{ gate='A0_TOP_EDGE_ONLY'; scenario='nominal'; topology='A0_TOP_EDGE_ONLY'; condition='A0_TOP_EDGE_ONLY'; density=1.0; thickness=0.002; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN },
    [pscustomobject]@{ gate='A1_TOP_EDGE_BOTTOM_TWO_CORNERS'; scenario='nominal'; topology='A1_TOP_EDGE_BOTTOM_TWO_CORNERS'; condition='A1_TOP_EDGE_BOTTOM_TWO_CORNERS'; density=1.0; thickness=0.002; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN },
    [pscustomobject]@{ gate='A2_FOUR_CORNERS_ONLY'; scenario='nominal'; topology='A2_FOUR_CORNERS_ONLY'; condition='A2_FOUR_CORNERS_ONLY'; density=1.0; thickness=0.002; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN },
    [pscustomobject]@{ gate='A3_FULL_PERIMETER_FIXED'; scenario='nominal'; topology='A3_FULL_PERIMETER_FIXED'; condition='A3_FULL_PERIMETER_FIXED'; density=1.0; thickness=0.002; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN },
    [pscustomobject]@{ gate='B_D225_NEW'; scenario='nominal'; topology='A0_TOP_EDGE_ONLY'; condition='B_D225_NEW'; density=2.25; thickness=0.002; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN },
    [pscustomobject]@{ gate='B_T003_NEW'; scenario='nominal'; topology='A0_TOP_EDGE_ONLY'; condition='B_T003_NEW'; density=1.0; thickness=0.003; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN },
    [pscustomobject]@{ gate='C_GAP02'; scenario='curtain_hidden_rigid'; topology='A0_TOP_EDGE_ONLY'; condition='C_GAP02'; density=1.0; thickness=0.002; hidden_x=0.890; hidden_y=[double]::NaN; hidden_width=[double]::NaN },
    [pscustomobject]@{ gate='C_GAP05'; scenario='curtain_hidden_rigid'; topology='A0_TOP_EDGE_ONLY'; condition='C_GAP05'; density=1.0; thickness=0.002; hidden_x=0.920; hidden_y=[double]::NaN; hidden_width=[double]::NaN },
    [pscustomobject]@{ gate='C_GAP08'; scenario='curtain_hidden_rigid'; topology='A0_TOP_EDGE_ONLY'; condition='C_GAP08'; density=1.0; thickness=0.002; hidden_x=0.950; hidden_y=[double]::NaN; hidden_width=[double]::NaN },
    [pscustomobject]@{ gate='B_D225_A1'; scenario='nominal'; topology='A1_TOP_EDGE_BOTTOM_TWO_CORNERS'; condition='B_D225_A1'; density=2.25; thickness=0.002; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN }
)

$results = @()
foreach ($condition in $conditions) {
    $common = @{
        GateId=$condition.gate; Scenario=$condition.scenario; Seed=$Seed; ConditionId=$condition.condition
        StructuralTopology=$condition.topology; SurfaceThickness=$condition.thickness
        HiddenRigidX=$condition.hidden_x; HiddenRigidY=$condition.hidden_y; HiddenRigidWidth=$condition.hidden_width
        DensityScaleMultiplier=$condition.density; Steps=$Steps
    }
    $noProbeArgs = $common.Clone(); $noProbeArgs.Mode='no_probe'; if ($DryRun) { $noProbeArgs.DryRun=$true }
    & $runner @noProbeArgs
    $noExit = $LASTEXITCODE
    $noSummary = Join-Path $workspace ("outputs\round4070_fixed10cm_judgment\physics\{0}\no_probe\summary.json" -f $condition.gate)
    $noStatus = if (Test-Path -LiteralPath $noSummary) { (Get-Content -LiteralPath $noSummary -Raw | ConvertFrom-Json).status } else { 'MISSING' }
    $probeExit = $null; $probeStatus = 'SKIPPED_NO_PROBE_GATE'
    if (-not $DryRun -and $noExit -eq 0 -and $noStatus -eq 'PASS_PHYSICS_SANITY') {
        $probeArgs = $common.Clone(); $probeArgs.Mode='probe'
        & $runner @probeArgs
        $probeExit = $LASTEXITCODE
        $probeSummary = Join-Path $workspace ("outputs\round4070_fixed10cm_judgment\physics\{0}\probe\summary.json" -f $condition.gate)
        if (Test-Path -LiteralPath $probeSummary) { $probeStatus = (Get-Content -LiteralPath $probeSummary -Raw | ConvertFrom-Json).status }
    } elseif ($DryRun) { $probeStatus = 'DRY_RUN' }
    $results += [pscustomobject]@{ gate_id=$condition.gate; topology=$condition.topology; no_probe_exit=$noExit; no_probe_status=$noStatus; probe_exit=$probeExit; probe_status=$probeStatus }
    [ordered]@{schema='lift-the-curtain.round4070-fixed10cm-physics-queue.v1'; updated_at=(Get-Date).ToString('o'); results=$results} |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $queueState -Encoding utf8
}
Write-Host ($results | ConvertTo-Csv -NoTypeInformation | Out-String)
if (@($results | Where-Object { $_.no_probe_status -notin @('PASS_PHYSICS_SANITY','DRY_RUN') -or $_.probe_status -notin @('CONTACT_PASS','PASS_PHYSICS_SANITY','SKIPPED_NO_PROBE_GATE','DRY_RUN') }).Count -gt 0) { exit 2 }
