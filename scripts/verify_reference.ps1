# Run each reference operator through jovin-harness inside WSL distro JovinHarness.
# Console lines are written as the WSL process starts. They are not held until the end.
$ErrorActionPreference = "Stop"
$env:JOVIN_HARNESS_RUNTIME = "wsl:JovinHarness"
if (-not $env:JOVIN_HARNESS_TIMEOUT_S) {
    $env:JOVIN_HARNESS_TIMEOUT_S = "30"
}
$root = Split-Path -Parent $PSScriptRoot
$exe = $env:JOVIN_HARNESS_EXE
if (-not $exe) {
    $discovered = Get-Command jovin-harness -ErrorAction SilentlyContinue
    $exe = if ($discovered) { $discovered.Source } else { "D:\JovinHarness_Artifacts\environments\windows\Scripts\jovin-harness.exe" }
}
$wsl = Join-Path $env:SystemRoot "System32\wsl.exe"
$logDir = Join-Path $root "reference\receipts"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$console = Join-Path $logDir "wsl-verify.log"
function Write-Console([string]$line) {
    $stamped = "{0} {1}" -f (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"), $line
    Write-Output $stamped
    Add-Content -Path $console -Encoding utf8 -Value $stamped
}
Write-Console "WSL initiate: $wsl --distribution JovinHarness --exec docker version"
& $wsl --distribution JovinHarness --exec docker version 2>&1 | ForEach-Object { Write-Console "wsl-docker: $_" }
if ($LASTEXITCODE -ne 0) {
    Write-Console "FAILED: WSL Docker preflight"
    exit 1
}
Write-Console "WSL initiate: $exe doctor"
$doctorOutput = & $exe doctor 2>&1 | Out-String
$doctorExit = $LASTEXITCODE
Write-Console "jovin-doctor: $doctorOutput"
$doctor = $null
try { $doctor = $doctorOutput | ConvertFrom-Json } catch { $doctor = $null }
if ($doctorExit -ne 0 -or $null -eq $doctor -or $doctor.ready -ne $true) {
    Write-Console "FAILED: jovin doctor is not ready"
    exit 1
}
$names = @(
    "boundary", "laplacian", "manifold", "fractional", "hypergraph",
    "resistance", "fiedler", "sparsify", "suzuki", "schedule", "composite", "symplectic"
)
$log = Join-Path $logDir "LOG.md"
if (-not (Test-Path $log)) {
    Set-Content -Path $log -Encoding utf8 -Value "# Receipts`r`n`r`nRows are copied from jovin-harness verify JSON. Runtime is wsl:JovinHarness.`r`n"
}
$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$block = @("", "## Run $stamp", "", "| module | status | passed | integrity | receipt_id | receipt_sha256 |", "|---|---|---|---|---|---|")
$failed = $false
foreach ($name in $names) {
    $code = Join-Path $root "reference\operators\$name.py"
    $tests = Join-Path $root "reference\tests\test_$name.py"
    Write-Console "WSL initiate verify: $name isolation=process timeout=30"
    $output = & $exe verify $code --tests $tests --isolation process --timeout 30 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    Write-Console "WSL verify returned: $name exit=$exitCode"
    Add-Content -Path (Join-Path $logDir "verify-$name.json") -Encoding utf8 -Value $output
    $parsed = $null
    try { $parsed = $output | ConvertFrom-Json } catch { $parsed = $null }
    if ($null -eq $parsed -or -not $parsed.passed -or $parsed.integrity -ne "process_isolated") {
        $failed = $true
        $status = if ($parsed) { [string]$parsed.status } else { "unparsed" }
        $block += "| $name | $status | false |  |  |  |"
        $block += ""
        $block += '```'
        $block += $output.Trim()
        $block += '```'
        Write-Console "CULLED: $name status=$status"
        continue
    }
    $block += "| $name | $($parsed.status) | $($parsed.passed) | $($parsed.integrity) | $($parsed.receipt_id) | $($parsed.receipt_sha256) |"
    Write-Console "passed: $name receipt=$($parsed.receipt_id)"
    if ($exitCode -ne 0) { $failed = $true }
}
Add-Content -Path $log -Encoding utf8 -Value ($block -join "`r`n")
Write-Console "WSL initiate: $exe receipts verify"
$chainOutput = & $exe receipts verify 2>&1 | Out-String
$chainExit = $LASTEXITCODE
Write-Console "jovin-chain: $chainOutput"
$chain = $null
try { $chain = $chainOutput | ConvertFrom-Json } catch { $chain = $null }
if ($chainExit -ne 0 -or $null -eq $chain -or $chain.valid -ne $true) {
    $failed = $true
    Write-Console "FAILED: jovin receipt chain is not valid"
}
if ($failed) {
    Write-Console "FAILED: one or more reference modules were culled"
    exit 1
}
Write-Console "DONE: reference operators passed process isolation"
exit 0
