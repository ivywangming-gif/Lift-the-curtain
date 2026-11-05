param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('split_free','split_left_limited','split_right_limited','strips_3_overlap','split_free_layered','split_left_limited_layered','split_right_limited_layered','strips_3_overlap_layered','split_free_layered24','split_left_limited_layered24','split_right_limited_layered24','strips_3_overlap_layered40')]
    [string]$Scene,
    [Parameter(Mandatory = $true)] [ValidateSet('no_probe','probe')] [string]$Mode,
    [int]$Seed = 47,
    [int]$OverallTimeoutSeconds = 600,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace 'outputs\round4070_structural_probe_discovery\physics\flexible'))
$runRoot = Join-Path $outputRoot ("{0}\{1}_seed{2}" -f $Scene,$Mode,$Seed)
if (-not $runRoot.StartsWith($outputRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Flexible scene path escapes C output root: $runRoot" }
if (Test-Path -LiteralPath (Join-Path $runRoot 'summary.json')) {
    Write-Host "SKIP_COMPLETE $Scene $Mode seed=$Seed"
    exit 0
}
if (Test-Path -LiteralPath $runRoot) {
    $existing = @(Get-ChildItem -LiteralPath $runRoot -Force -ErrorAction SilentlyContinue)
    if ($existing.Count -gt 0) { throw "Incomplete validation directory is preserved: $runRoot" }
}
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$scriptArgs = @('--scene',$Scene,'--mode',$Mode,'--seed',[string]$Seed,'--run-root',$runRoot,'--headless')
if ($Mode -eq 'probe') {
    $control = Join-Path $outputRoot ("{0}\no_probe_seed{1}\summary.json" -f $Scene,$Seed)
    if (-not (Test-Path -LiteralPath $control)) { throw "Matched no-probe summary is required first: $control" }
    $scriptArgs += @('--matched-no-probe-summary',$control)
}
[ordered]@{script='scripts/round4070_structural_probe_discovery/validate_flexible_scene.py';args=$scriptArgs;dry_run=[bool]$DryRun;model_contract='ARTICULATED_STRIP_SURROGATE'} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runRoot 'command.json') -Encoding utf8
if ($DryRun) { $scriptArgs -join ' '; exit 0 }
$guard = Join-Path $PSScriptRoot 'run_guarded_with_resources.ps1'
$script = Join-Path $PSScriptRoot 'validate_flexible_scene.py'
& $guard -ScriptPath $script -RunRoot $runRoot -ScriptArgs $scriptArgs -OverallTimeoutSeconds $OverallTimeoutSeconds
exit $LASTEXITCODE
