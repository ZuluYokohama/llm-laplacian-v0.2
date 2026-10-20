# Push a ledger line when a new Arm 0 weight file has finished writing.
# Weight files are not added. This process is separate from the training queue.
$ErrorActionPreference = "Continue"
Set-Location "D:\LLM_LAPLACIAN_TEST_V0.2"
while ($true) {
    & "D:\LLM_LAPLACIAN_TEST_V0.2\.venv\Scripts\python.exe" "D:\LLM_LAPLACIAN_TEST_V0.2\scripts\publish_epoch.py" --publish
    Start-Sleep -Seconds 60
}
