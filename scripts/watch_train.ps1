# Watch one Laplacian queue process.
# Stdout is only ACTION_REQUIRED, FAILED, or DONE. Diagnostics go to a pid-keyed file.
param(
    [Parameter(Mandatory = $true)][int]$TargetPid,
    [string]$Root = "D:\LLM_LAPLACIAN_TEST_V0.2"
)

$ErrorActionPreference = "Continue"
$diagDir = "C:\Users\JT-DEV1\.grok\long-running-background-tasks"
New-Item -ItemType Directory -Force -Path $diagDir | Out-Null
$diag = Join-Path $diagDir ("watch_laplacian_{0}.log" -f $TargetPid)
$queueLog = Join-Path $Root "runs\queue.log"
$heartbeat = Join-Path $Root "runs\heartbeat.txt"
$doneFlag = Join-Path $Root "runs\QUEUE_DONE"
$guardMib = 11.2 * 1024
$offset = 0
if (Test-Path $queueLog) {
    $offset = (Get-Item $queueLog).Length
}
$stallEpisode = $false
$vramEpisode = $false

function Write-Diag([string]$line) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Path $diag -Encoding utf8 -Value ("{0} {1}" -f $stamp, $line)
}

function Get-NewestUtc {
    $times = @()
    foreach ($path in @($queueLog, $heartbeat)) {
        if (Test-Path $path) {
            $times += (Get-Item $path).LastWriteTimeUtc
        }
    }
    if ($times.Count -eq 0) {
        return $null
    }
    return ($times | Measure-Object -Maximum).Maximum
}

function Get-5070Mib {
    $raw = & nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits 2>$null
    if (-not $raw) {
        return $null
    }
    foreach ($row in @($raw)) {
        if ($row -like "*5070*") {
            $parts = $row.Split(",")
            return [double]$parts[$parts.Length - 1].Trim()
        }
    }
    return $null
}

function Read-NewLog {
    if (-not (Test-Path $queueLog)) {
        return ""
    }
    $stream = New-Object System.IO.FileStream($queueLog, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
        if ($script:offset -gt $stream.Length) {
            $script:offset = 0
        }
        [void]$stream.Seek($script:offset, [System.IO.SeekOrigin]::Begin)
        $count = [int][Math]::Min($stream.Length - $script:offset, 1048576)
        if ($count -le 0) {
            return ""
        }
        $buffer = New-Object byte[] $count
        $read = $stream.Read($buffer, 0, $count)
        $script:offset += $read
        return [System.Text.Encoding]::UTF8.GetString($buffer, 0, $read)
    }
    finally {
        $stream.Close()
    }
}

Write-Diag ("watch start pid={0}" -f $TargetPid)
while ($true) {
    $alive = $null -ne (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue)
    if (-not $alive) {
        if (Test-Path $doneFlag) {
            Write-Output ("DONE: queue pid {0} exited and runs/QUEUE_DONE exists" -f $TargetPid)
            exit 0
        }
        Write-Diag "pid dead without QUEUE_DONE"
        Write-Output ("FAILED: queue pid {0} exited without runs/QUEUE_DONE" -f $TargetPid)
        exit 1
    }
    $fresh = Read-NewLog
    if ($fresh -match "nan=1") {
        Write-Diag "nan=1"
        Write-Output ("FAILED: nan=1 in queue log for pid {0}" -f $TargetPid)
        exit 1
    }
    if ($fresh -match "residual=nan") {
        Write-Diag "cg residual nan"
        Write-Output ("FAILED: CG residual nan in queue log for pid {0}" -f $TargetPid)
        exit 1
    }
    if ($fresh -match "OutOfMemoryError|CUDA out of memory") {
        Write-Diag "cuda oom"
        Write-Output ("FAILED: CUDA OOM in queue log for pid {0}" -f $TargetPid)
        exit 1
    }
    $used = Get-5070Mib
    if ($null -eq $used) {
        Write-Diag "nvidia-smi returned no 5070 memory reading"
    }
    elseif ($used -ge $guardMib) {
        if (-not $vramEpisode) {
            $vramEpisode = $true
            Write-Output ("ACTION_REQUIRED: RTX 5070 VRAM {0} MiB is at or above the 11.2 GB guard while pid {1} is alive" -f $used, $TargetPid)
        }
    }
    else {
        $vramEpisode = $false
    }
    $newest = Get-NewestUtc
    if ($null -eq $newest) {
        $age = 999999
    }
    else {
        $age = ((Get-Date).ToUniversalTime() - $newest).TotalSeconds
    }
    if ($age -gt 600) {
        if (-not $stallEpisode) {
            $stallEpisode = $true
            Write-Diag ("stale age={0}" -f [int]$age)
            Write-Output ("ACTION_REQUIRED: queue log and heartbeat stale for {0}s while pid {1} is alive" -f [int]$age, $TargetPid)
        }
    }
    else {
        $stallEpisode = $false
    }
    Start-Sleep -Seconds 30
}
