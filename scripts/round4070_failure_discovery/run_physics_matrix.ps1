param([int]$Seed = 61, [int]$Steps = 720, [switch]$DryRun)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$runner = Join-Path $PSScriptRoot 'run_physics_gate.ps1'
$conditions = @(
    [pscustomobject]@{ gate='FABRIC_ONLY'; scenario='nominal'; condition='FABRIC_ONLY'; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=1.0; damping=[double]::NaN; bend=[double]::NaN; length=2.0; top=[double]::NaN },
    [pscustomobject]@{ gate='S1B_NEAR'; scenario='curtain_hidden_rigid'; condition='S1B_NEAR'; hidden_x=0.916; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=1.0; damping=[double]::NaN; bend=[double]::NaN; length=2.0; top=[double]::NaN },
    [pscustomobject]@{ gate='S1C_FAR'; scenario='curtain_hidden_rigid'; condition='S1C_FAR'; hidden_x=1.240; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=1.0; damping=[double]::NaN; bend=[double]::NaN; length=2.0; top=[double]::NaN },
    [pscustomobject]@{ gate='S1D_PARTIAL'; scenario='curtain_hidden_rigid'; condition='S1D_PARTIAL'; hidden_x=0.916; hidden_y=0.420; hidden_width=0.500; density=1.0; damping=[double]::NaN; bend=[double]::NaN; length=2.0; top=[double]::NaN },
    [pscustomobject]@{ gate='S2_D125'; scenario='nominal'; condition='S2_D125'; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=1.25; damping=[double]::NaN; bend=[double]::NaN; length=2.0; top=[double]::NaN },
    [pscustomobject]@{ gate='S2_D150'; scenario='nominal'; condition='S2_D150'; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=1.50; damping=[double]::NaN; bend=[double]::NaN; length=2.0; top=[double]::NaN },
    [pscustomobject]@{ gate='S2_D200'; scenario='nominal'; condition='S2_D200'; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=2.00; damping=[double]::NaN; bend=[double]::NaN; length=2.0; top=[double]::NaN },
    [pscustomobject]@{ gate='S2_DAMP'; scenario='nominal'; condition='S2_DAMP'; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=1.0; damping=0.20; bend=[double]::NaN; length=2.0; top=[double]::NaN },
    [pscustomobject]@{ gate='S2_BEND'; scenario='nominal'; condition='S2_BEND'; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=1.0; damping=[double]::NaN; bend=0.08; length=2.0; top=[double]::NaN },
    [pscustomobject]@{ gate='S3_BOTTOM_LEFT'; scenario='bottom_left_anchor'; condition='S3_BOTTOM_LEFT'; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=1.0; damping=[double]::NaN; bend=[double]::NaN; length=2.0; top=[double]::NaN },
    [pscustomobject]@{ gate='S3_BOTTOM_RIGHT'; scenario='bottom_right_anchor'; condition='S3_BOTTOM_RIGHT'; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=1.0; damping=[double]::NaN; bend=[double]::NaN; length=2.0; top=[double]::NaN },
    [pscustomobject]@{ gate='S4_LEN180'; scenario='nominal'; condition='S4_LEN180'; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=1.0; damping=[double]::NaN; bend=[double]::NaN; length=1.80; top=1.82 },
    [pscustomobject]@{ gate='S4_LEN160'; scenario='nominal'; condition='S4_LEN160'; hidden_x=[double]::NaN; hidden_y=[double]::NaN; hidden_width=[double]::NaN; density=1.0; damping=[double]::NaN; bend=[double]::NaN; length=1.60; top=1.62 }
)

$results = @()
foreach ($condition in $conditions) {
    $common = @{
        GateId=$condition.gate; Scenario=$condition.scenario; Seed=$Seed; ConditionId=$condition.condition
        HiddenRigidX=$condition.hidden_x; HiddenRigidY=$condition.hidden_y; HiddenRigidWidth=$condition.hidden_width
        DensityScaleMultiplier=$condition.density; CurtainDamping=$condition.damping; CurtainBendStiffness=$condition.bend
        CurtainLength=$condition.length; CurtainTopZ=$condition.top; Steps=$Steps
    }
    $noProbeArgs = $common.Clone(); $noProbeArgs.Mode='no_probe'; if ($DryRun) { $noProbeArgs.DryRun=$true }
    & $runner @noProbeArgs
    $noExit = $LASTEXITCODE
    $noSummary = Join-Path $workspace ("outputs\round4070_failure_discovery\physics\{0}\no_probe\summary.json" -f $condition.gate)
    $noStatus = if (Test-Path -LiteralPath $noSummary) { (Get-Content -LiteralPath $noSummary -Raw | ConvertFrom-Json).status } else { 'MISSING' }
    $probeExit = $null; $probeStatus = 'SKIPPED_NO_PROBE_GATE'
    if (-not $DryRun -and $noExit -eq 0 -and $noStatus -eq 'PASS_PHYSICS_SANITY') {
        $probeArgs = $common.Clone(); $probeArgs.Mode='probe'
        & $runner @probeArgs
        $probeExit = $LASTEXITCODE
        $probeSummary = Join-Path $workspace ("outputs\round4070_failure_discovery\physics\{0}\probe\summary.json" -f $condition.gate)
        if (Test-Path -LiteralPath $probeSummary) { $probeStatus = (Get-Content -LiteralPath $probeSummary -Raw | ConvertFrom-Json).status }
    } elseif ($DryRun) { $probeStatus = 'DRY_RUN' }
    $results += [pscustomobject]@{ gate_id=$condition.gate; no_probe_exit=$noExit; no_probe_status=$noStatus; probe_exit=$probeExit; probe_status=$probeStatus }
    [ordered]@{updated_at=(Get-Date).ToString('o'); last_gate=$condition.gate; results=$results} |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $workspace 'outputs\round4070_failure_discovery\physics_queue_state.json') -Encoding utf8
}
Write-Host ($results | ConvertTo-Csv -NoTypeInformation | Out-String)
if (@($results | Where-Object { $_.no_probe_status -notin @('PASS_PHYSICS_SANITY','DRY_RUN') -or $_.probe_status -notin @('CONTACT_PASS','PASS_PHYSICS_SANITY','SKIPPED_NO_PROBE_GATE','DRY_RUN') }).Count -gt 0) { exit 2 }
