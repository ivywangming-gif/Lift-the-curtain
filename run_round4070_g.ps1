param(
    [ValidateSet('all','physics','g1','status','analyze','train')] [string]$Mode='all',
    [string]$Scene='',
    [int]$MaxJobs=0,
    [switch]$RetryRuntimeFailures,
    [int]$ChunkJobs=12,
    [int]$MaxFits=0
)
$ErrorActionPreference='Stop'
$manifest=Join-Path $PSScriptRoot 'outputs\round4070_comprehensive_traversability\campaign_manifest_v3.json'
if (-not (Test-Path -LiteralPath $manifest)) {
    & 'C:\conda_envs\isaaclab\python.exe' (Join-Path $PSScriptRoot 'scripts\round4070_g\physics_restore_manifest.py')
    if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
}
if ($Mode -in @('all','analyze','train')) {
    $runner=Join-Path $PSScriptRoot 'scripts\round4070_g\workflow.py'
    $arguments=@($runner,'--mode',$Mode,'--max-jobs',[string]$MaxJobs,'--chunk-jobs',[string]$ChunkJobs)
    if ($MaxFits -gt 0) {$arguments+=@('--max-fits',[string]$MaxFits)}
} else {
    $runner=Join-Path $PSScriptRoot 'scripts\round4070_g\physics_campaign.py'
    $arguments=@($runner,'--mode',$Mode,'--max-jobs',[string]$MaxJobs)
}
if ($Scene) {$arguments+=@('--scene',$Scene)}
if ($RetryRuntimeFailures) {$arguments+='--retry-runtime-failures'}
& 'C:\conda_envs\isaaclab\python.exe' @arguments
exit $LASTEXITCODE
