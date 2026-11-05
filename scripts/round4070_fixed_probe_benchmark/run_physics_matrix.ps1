param(
    [int]$Seed = 4070,
    [int]$Steps = 720,
    [int]$OverallTimeoutSeconds = 900,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$runner = Join-Path $PSScriptRoot 'run_physics_gate.ps1'
$outputRoot = Join-Path $workspace 'outputs\round4070_fixed_probe_benchmark\physics'
$queueState = Join-Path $workspace 'outputs\round4070_fixed_probe_benchmark\physics_queue_state.json'
$statusPath = Join-Path $workspace 'configs\round4070_fixed_probe_benchmark\PHYSICS_GATE_STATUS.json'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$conditions = @(
    [pscustomobject]@{ gate='A0_TOP_EDGE_ONLY'; scenario='nominal'; topology='A0_TOP_EDGE_ONLY'; condition='FABRIC_NOMINAL'; density=1.0; thickness=0.002; hidden_x=[double]::NaN },
    [pscustomobject]@{ gate='A1_BOTTOM_LEFT_ANCHOR'; scenario='bottom_left_anchor'; topology='A0_TOP_EDGE_ONLY'; condition='FABRIC_BOTTOM_LEFT_ANCHOR'; density=1.0; thickness=0.002; hidden_x=[double]::NaN },
    [pscustomobject]@{ gate='A1_BOTTOM_RIGHT_ANCHOR'; scenario='bottom_right_anchor'; topology='A0_TOP_EDGE_ONLY'; condition='FABRIC_BOTTOM_RIGHT_ANCHOR'; density=1.0; thickness=0.002; hidden_x=[double]::NaN },
    [pscustomobject]@{ gate='A1_TOP_EDGE_BOTTOM_TWO_CORNERS'; scenario='both_bottom_corners'; topology='A1_TOP_EDGE_BOTTOM_TWO_CORNERS'; condition='FABRIC_BOTH_BOTTOM_CORNERS'; density=1.0; thickness=0.002; hidden_x=[double]::NaN },
    [pscustomobject]@{ gate='A2_FOUR_CORNERS_ONLY'; scenario='nominal'; topology='A2_FOUR_CORNERS_ONLY'; condition='FABRIC_FOUR_CORNERS_ONLY'; density=1.0; thickness=0.002; hidden_x=[double]::NaN },
    [pscustomobject]@{ gate='A3_FULL_PERIMETER_FIXED'; scenario='nominal'; topology='A3_FULL_PERIMETER_FIXED'; condition='FABRIC_FULL_PERIMETER_FIXED'; density=1.0; thickness=0.002; hidden_x=[double]::NaN },
    [pscustomobject]@{ gate='B_D225_NEW'; scenario='nominal'; topology='A0_TOP_EDGE_ONLY'; condition='FABRIC_HEAVY_FREE'; density=2.25; thickness=0.002; hidden_x=[double]::NaN },
    [pscustomobject]@{ gate='B_D225_A1'; scenario='nominal'; topology='A1_TOP_EDGE_BOTTOM_TWO_CORNERS'; condition='FABRIC_HEAVY_CONSTRAINED'; density=2.25; thickness=0.002; hidden_x=[double]::NaN },
    [pscustomobject]@{ gate='C_GAP02'; scenario='curtain_hidden_rigid'; topology='A0_TOP_EDGE_ONLY'; condition='FABRIC_WALL_GAP_02'; density=1.0; thickness=0.002; hidden_x=0.890 },
    [pscustomobject]@{ gate='C_GAP05'; scenario='curtain_hidden_rigid'; topology='A0_TOP_EDGE_ONLY'; condition='FABRIC_WALL_GAP_05'; density=1.0; thickness=0.002; hidden_x=0.920 },
    [pscustomobject]@{ gate='C_GAP08'; scenario='curtain_hidden_rigid'; topology='A0_TOP_EDGE_ONLY'; condition='FABRIC_WALL_GAP_08'; density=1.0; thickness=0.002; hidden_x=0.950 }
)

$results = @()
foreach ($condition in $conditions) {
    $common = @{
        GateId=$condition.gate; Scenario=$condition.scenario; Seed=$Seed; ConditionId=$condition.condition
        StructuralTopology=$condition.topology; SurfaceThickness=$condition.thickness
        HiddenRigidX=$condition.hidden_x; DensityScaleMultiplier=$condition.density
        Steps=$Steps; OverallTimeoutSeconds=$OverallTimeoutSeconds
    }
    $noArgs = $common.Clone(); $noArgs.Mode='no_probe'; if ($DryRun) { $noArgs.DryRun=$true }
    & $runner @noArgs
    $noExit = $LASTEXITCODE
    $noSummary = Join-Path $outputRoot ("{0}\no_probe\summary.json" -f $condition.gate)
    $noStatus = if (Test-Path -LiteralPath $noSummary) { [string](Get-Content -LiteralPath $noSummary -Raw | ConvertFrom-Json).status } elseif ($DryRun) { 'DRY_RUN' } else { 'MISSING' }
    $probeExit = $null; $probeStatus = 'SKIPPED_NO_PROBE_GATE'
    if (-not $DryRun -and $noExit -eq 0 -and $noStatus -eq 'PASS_PHYSICS_SANITY') {
        $probeArgs = $common.Clone(); $probeArgs.Mode='probe'
        & $runner @probeArgs
        $probeExit = $LASTEXITCODE
        $probeSummary = Join-Path $outputRoot ("{0}\probe\summary.json" -f $condition.gate)
        if (Test-Path -LiteralPath $probeSummary) { $probeStatus = [string](Get-Content -LiteralPath $probeSummary -Raw | ConvertFrom-Json).status }
    } elseif ($DryRun) { $probeStatus = 'DRY_RUN' }
    $results += [pscustomobject]@{
        gate_id=$condition.gate; condition_id=$condition.condition; topology=$condition.topology
        no_probe_exit=$noExit; no_probe_status=$noStatus; probe_exit=$probeExit; probe_status=$probeStatus
        run_root=(Join-Path $outputRoot $condition.gate); finished_at=(Get-Date).ToString('o')
    }
    [ordered]@{schema='lift-the-curtain.round4070-fixed-probe-physics-queue.v1'; updated_at=(Get-Date).ToString('o'); results=$results} |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $queueState -Encoding utf8
}

$gates = [ordered]@{}
foreach ($result in $results) {
    $pass = $result.no_probe_status -eq 'PASS_PHYSICS_SANITY' -and $result.probe_status -in @('CONTACT_PASS', 'PASS_PHYSICS_SANITY')
    $gates[$result.gate_id] = [ordered]@{
        status=if ($DryRun) { 'DRY_RUN' } elseif ($pass) { 'PASS_E_PHYSICS' } else { 'BLOCKED' }
        condition_id=$result.condition_id; no_probe_status=$result.no_probe_status; probe_status=$result.probe_status
        evidence_root=$result.run_root; model_contract='SINGLE_SURFACE_PHYSICS_GATE'
    }
}
$gates['PASS_CONTROL'] = [ordered]@{ status='PASS_CONTROL'; condition_id='EMPTY_CONTROL_OR_RIGID_CONTROL'; evidence_root='G1_PHASE_0'; model_contract='G1_RUNTIME_CONTROL' }
$status = [ordered]@{
    schema='lift-the-curtain.round4070-fixed-probe-physics-gate-status.v1'
    campaign='Round 4070-E'
    updated_at=(Get-Date).ToString('o')
    gates=$gates
    p0_claim_scope='Only P0 single-sheet scenes are covered here; P1 multi-sheet attempts are reported separately.'
}
$status | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $statusPath -Encoding utf8
Write-Host ($results | ConvertTo-Csv -NoTypeInformation | Out-String)
if (-not $DryRun -and @($results | Where-Object { $_.no_probe_status -ne 'PASS_PHYSICS_SANITY' -or $_.probe_status -notin @('CONTACT_PASS','PASS_PHYSICS_SANITY') }).Count -gt 0) { exit 2 }
