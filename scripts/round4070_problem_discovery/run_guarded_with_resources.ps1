param(
    [Parameter(Mandatory = $true)] [string]$ScriptPath,
    [Parameter(Mandatory = $true)] [string]$RunRoot,
    [string[]]$ScriptArgs = @(),
    [int]$OverallTimeoutSeconds = 1200,
    [int]$PollSeconds = 2,
    [string]$PythonExe = 'C:\conda_envs\isaaclab\python.exe'
)

$ErrorActionPreference = 'Stop'
$resolvedRunRoot = [System.IO.Path]::GetFullPath($RunRoot)
$resolvedScript = [System.IO.Path]::GetFullPath($ScriptPath)
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $resolvedRunRoot.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "RunRoot must stay inside the workspace: $resolvedRunRoot"
}
if (-not (Test-Path -LiteralPath $resolvedScript -PathType Leaf)) {
    throw "Script not found: $resolvedScript"
}
if ($OverallTimeoutSeconds -le 0 -or $PollSeconds -le 0) { throw 'Timeout and poll values must be positive.' }

New-Item -ItemType Directory -Path $resolvedRunRoot -Force | Out-Null
$stdoutPath = Join-Path $resolvedRunRoot 'process_stdout.log'
$stderrPath = Join-Path $resolvedRunRoot 'process_stderr.log'
$heartbeatPath = Join-Path $resolvedRunRoot 'stage_heartbeat.jsonl'
$watchdogPath = Join-Path $resolvedRunRoot 'watchdog_summary.json'
$resourcePath = Join-Path $resolvedRunRoot 'resource_telemetry.csv'
$resourceSummaryPath = Join-Path $resolvedRunRoot 'resource_summary.json'
$resourceErrorPath = Join-Path $resolvedRunRoot 'resource_sampling_errors.log'
$phaseTimeouts = @{ BEFORE_APP=180; AFTER_APP=180; AFTER_SCENE=120; AFTER_RESET=120; FIRST_PHYSICS_STEP=120; ROLLOUT_PROGRESS=90; FINALIZE=30 }

try {
    Add-Type -AssemblyName Microsoft.VisualBasic -ErrorAction Stop
    $computerInfo = [Microsoft.VisualBasic.Devices.ComputerInfo]::new()
    $totalPhysicalMemoryMB = [math]::Round($computerInfo.TotalPhysicalMemory / 1MB, 1)
} catch {
    $computerInfo = $null; $totalPhysicalMemoryMB = $null
    $_ | Out-String | Add-Content -LiteralPath $resourceErrorPath -Encoding utf8
}
try {
    $systemCpuCounter = [System.Diagnostics.PerformanceCounter]::new('Processor', '% Processor Time', '_Total')
    $null = $systemCpuCounter.NextValue()
} catch {
    $systemCpuCounter = $null
    $_ | Out-String | Add-Content -LiteralPath $resourceErrorPath -Encoding utf8
}

function Get-GpuSample {
    try {
        $raw = & nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if (-not $raw) { return @{} }
        $parts = @($raw -split ',' | ForEach-Object { $_.Trim() })
        if ($parts.Count -lt 9) { return @{} }
        return @{ name=$parts[0]; driver=$parts[1]; memory_total_mb=$parts[2]; memory_used_mb=$parts[3]; utilization_gpu_pct=$parts[4]; utilization_memory_pct=$parts[5]; temperature_c=$parts[6]; power_w=$parts[7]; clock_sm_mhz=$parts[8] }
    } catch {
        $_ | Out-String | Add-Content -LiteralPath $resourceErrorPath -Encoding utf8
        return @{}
    }
}

function Convert-NumberOrNull([object]$Value) {
    if ($null -eq $Value) { return $null }
    $parsed = 0.0
    if ([double]::TryParse([string]$Value, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) { return $parsed }
    return $null
}

$arguments = @($resolvedScript) + $ScriptArgs
$startedAt = Get-Date
$process = Start-Process -FilePath $PythonExe -ArgumentList $arguments -WorkingDirectory $workspace -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -WindowStyle Hidden -PassThru
$lastStage = $null; $lastStageStatus = $null; $lastStageWallTime = 0.0; $timeoutReason = $null
$lastCpuSeconds = 0.0; $lastSampleAt = $startedAt; $sampleIndex = 0

while (-not $process.HasExited) {
    Start-Sleep -Seconds $PollSeconds
    $now = Get-Date; $elapsed = ($now - $startedAt).TotalSeconds
    try { $process.Refresh() } catch {}
    if (Test-Path -LiteralPath $heartbeatPath) {
        $lastLine = Get-Content -LiteralPath $heartbeatPath -Tail 1 -ErrorAction SilentlyContinue
        if ($lastLine) {
            try {
                $event = $lastLine | ConvertFrom-Json
                $lastStage = [string]$event.stage; $lastStageStatus = [string]$event.status; $lastStageWallTime = [double]$event.wall_time_s
            } catch {}
        }
    }
    $processCpuPercent = $null; $workingSetMB = $null; $privateMB = $null
    try {
        $cpuSeconds = $process.TotalProcessorTime.TotalSeconds
        $sampleDuration = [math]::Max(0.001, ($now - $lastSampleAt).TotalSeconds)
        $processCpuPercent = 100.0 * ($cpuSeconds - $lastCpuSeconds) / $sampleDuration / [Environment]::ProcessorCount
        $lastCpuSeconds = $cpuSeconds
        $workingSetMB = $process.WorkingSet64 / 1MB; $privateMB = $process.PrivateMemorySize64 / 1MB
    } catch {}
    $lastSampleAt = $now
    $availableMemoryMB = $null
    if ($computerInfo) { try { $availableMemoryMB = $computerInfo.AvailablePhysicalMemory / 1MB } catch {} }
    $systemCpuPercent = $null
    if ($systemCpuCounter) { try { $systemCpuPercent = $systemCpuCounter.NextValue() } catch {} }
    $gpu = Get-GpuSample
    $sample = [pscustomobject][ordered]@{
        sample_index=$sampleIndex; timestamp=$now.ToString('o'); elapsed_s=[math]::Round($elapsed,3); pid=$process.Id
        process_cpu_percent=if($null -eq $processCpuPercent){$null}else{[math]::Round($processCpuPercent,3)}
        process_working_set_mb=if($null -eq $workingSetMB){$null}else{[math]::Round($workingSetMB,3)}
        process_private_mb=if($null -eq $privateMB){$null}else{[math]::Round($privateMB,3)}
        system_cpu_percent=if($null -eq $systemCpuPercent){$null}else{[math]::Round($systemCpuPercent,3)}
        system_available_ram_mb=if($null -eq $availableMemoryMB){$null}else{[math]::Round($availableMemoryMB,3)}
        system_total_ram_mb=$totalPhysicalMemoryMB
        gpu_name=$gpu.name; gpu_driver=$gpu.driver
        gpu_memory_total_mb=Convert-NumberOrNull $gpu.memory_total_mb; gpu_memory_used_mb=Convert-NumberOrNull $gpu.memory_used_mb
        gpu_utilization_percent=Convert-NumberOrNull $gpu.utilization_gpu_pct; gpu_memory_utilization_percent=Convert-NumberOrNull $gpu.utilization_memory_pct
        gpu_temperature_c=Convert-NumberOrNull $gpu.temperature_c; gpu_power_w=Convert-NumberOrNull $gpu.power_w; gpu_sm_clock_mhz=Convert-NumberOrNull $gpu.clock_sm_mhz
        stage=$lastStage; stage_status=$lastStageStatus; stage_wall_time_s=$lastStageWallTime
    }
    $sample | Export-Csv -LiteralPath $resourcePath -NoTypeInformation -Append -Encoding utf8
    $sampleIndex += 1
    if ($elapsed -gt $OverallTimeoutSeconds) { $timeoutReason = "OVERALL_TIMEOUT_${OverallTimeoutSeconds}s" }
    elseif ($lastStage -and $phaseTimeouts.ContainsKey($lastStage)) {
        $limit = [double]$phaseTimeouts[$lastStage]
        if (($elapsed - $lastStageWallTime) -gt $limit) {
            $timeoutReason = if($lastStage -eq 'FINALIZE'){"CLEANUP_TIMEOUT_AFTER_FINALIZE_${limit}s"}else{"PHASE_TIMEOUT_${lastStage}_${limit}s"}
        }
    }
    if ($timeoutReason) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue; break }
}

$process.WaitForExit(); $finishedAt = Get-Date
$heartbeatEvents = @()
if (Test-Path -LiteralPath $heartbeatPath) {
    foreach ($line in Get-Content -LiteralPath $heartbeatPath -ErrorAction SilentlyContinue) { try { $heartbeatEvents += ($line | ConvertFrom-Json) } catch {} }
}
$firstPhysics = $heartbeatEvents | Where-Object { $_.stage -eq 'FIRST_PHYSICS_STEP' } | Select-Object -First 1
$finalize = $heartbeatEvents | Where-Object { $_.stage -eq 'FINALIZE' } | Select-Object -Last 1
$telemetryPath = Join-Path $resolvedRunRoot 'telemetry.csv'
if (-not (Test-Path -LiteralPath $telemetryPath)) { $telemetryPath = Join-Path $resolvedRunRoot 'telemetry_partial.csv' }
$lastTelemetry = $null
if (Test-Path -LiteralPath $telemetryPath) { try { $lastTelemetry = Import-Csv -LiteralPath $telemetryPath | Select-Object -Last 1 } catch {} }
$simulatedTimeS = if($lastTelemetry){Convert-NumberOrNull $lastTelemetry.time_s}else{$null}
$maxStep = if($lastTelemetry){Convert-NumberOrNull $lastTelemetry.step}else{$null}
$rolloutWallS = if($firstPhysics -and $finalize){[double]$finalize.wall_time_s - [double]$firstPhysics.wall_time_s}else{$null}
$resourceRows = @()
if (Test-Path -LiteralPath $resourcePath) { try { $resourceRows = @(Import-Csv -LiteralPath $resourcePath) } catch {} }
function Measure-Resource([string]$Property,[string]$Mode) {
    $values = @($resourceRows | ForEach-Object { Convert-NumberOrNull $_.$Property } | Where-Object { $null -ne $_ })
    if (-not $values) { return $null }
    if ($Mode -eq 'Max') { return ($values | Measure-Object -Maximum).Maximum }
    if ($Mode -eq 'Min') { return ($values | Measure-Object -Minimum).Minimum }
    return ($values | Measure-Object -Average).Average
}
$resourceSummary = [ordered]@{
    schema='lift-the-curtain.round4070-problem-discovery-resource-summary.v1'; sample_count=$resourceRows.Count
    gpu_name=if($resourceRows){$resourceRows[-1].gpu_name}else{$null}; gpu_driver=if($resourceRows){$resourceRows[-1].gpu_driver}else{$null}
    gpu_memory_total_mb=Measure-Resource 'gpu_memory_total_mb' 'Max'; gpu_memory_peak_mb=Measure-Resource 'gpu_memory_used_mb' 'Max'
    gpu_utilization_mean_percent=Measure-Resource 'gpu_utilization_percent' 'Mean'; gpu_utilization_peak_percent=Measure-Resource 'gpu_utilization_percent' 'Max'
    gpu_temperature_peak_c=Measure-Resource 'gpu_temperature_c' 'Max'; gpu_power_peak_w=Measure-Resource 'gpu_power_w' 'Max'
    process_working_set_peak_mb=Measure-Resource 'process_working_set_mb' 'Max'; process_private_peak_mb=Measure-Resource 'process_private_mb' 'Max'
    process_cpu_mean_percent=Measure-Resource 'process_cpu_percent' 'Mean'; system_cpu_mean_percent=Measure-Resource 'system_cpu_percent' 'Mean'
    system_ram_total_mb=$totalPhysicalMemoryMB; system_ram_available_min_mb=Measure-Resource 'system_available_ram_mb' 'Min'
    isaac_initialization_time_s=if($firstPhysics){[double]$firstPhysics.wall_time_s}else{$null}; rollout_wall_time_s=$rolloutWallS
    simulated_time_s=$simulatedTimeS; physics_steps=$maxStep
    physics_steps_per_wall_second=if($rolloutWallS -and $maxStep){$maxStep/$rolloutWallS}else{$null}
    simulated_to_wall_ratio=if($rolloutWallS -and $simulatedTimeS){$simulatedTimeS/$rolloutWallS}else{$null}
    timeout_reason=$timeoutReason; last_stage=$lastStage; last_stage_status=$lastStageStatus
    resource_sampling_errors=if(Test-Path -LiteralPath $resourceErrorPath){$resourceErrorPath}else{$null}
}
$resourceSummary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resourceSummaryPath -Encoding utf8
$watchdogSummary = [ordered]@{
    script=$resolvedScript; run_root=$resolvedRunRoot; pid_started_by_round4070_problem_discovery=$process.Id
    started_at=$startedAt.ToString('o'); finished_at=$finishedAt.ToString('o'); wall_time_s=($finishedAt-$startedAt).TotalSeconds
    exit_code=if($timeoutReason){$null}else{$process.ExitCode}; timeout_reason=$timeoutReason; last_valid_stage=$lastStage; last_stage_status=$lastStageStatus
    heartbeat_path=$heartbeatPath; stdout_path=$stdoutPath; stderr_path=$stderrPath; resource_telemetry_path=$resourcePath; resource_summary_path=$resourceSummaryPath
    cleanup_scope='Only the explicitly started process id was stopped.'
}
$watchdogSummary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $watchdogPath -Encoding utf8
$watchdogSummary | ConvertTo-Json -Depth 6
if ($timeoutReason -and $timeoutReason.StartsWith('CLEANUP_TIMEOUT_AFTER_FINALIZE') -and $lastStageStatus -in @('PASS','FAIL')) { exit 0 }
if ($timeoutReason) { exit 124 }
exit $process.ExitCode

