# Detached serial-queue resume. The tool shell is killed after 10 hours;
# this process is started outside that shell so Arm 0 can keep last.pt.
$ErrorActionPreference = "Stop"
Set-Location "D:\LLM_LAPLACIAN_TEST_V0.2"
$env:PYTHONPATH = "D:\LLM_LAPLACIAN_TEST_V0.2\src"
$env:TEMP = "D:\LLM_LAPLACIAN_TEST_V0.2\cache\tmp"
$env:TMP = "D:\LLM_LAPLACIAN_TEST_V0.2\cache\tmp"
$env:HF_HOME = "D:\LLM_LAPLACIAN_TEST_V0.2\cache\hf"
$env:HF_DATASETS_CACHE = "D:\LLM_LAPLACIAN_TEST_V0.2\cache\hf\datasets"
$env:TORCH_HOME = "D:\LLM_LAPLACIAN_TEST_V0.2\cache\torch"
$env:PIP_CACHE_DIR = "D:\LLM_LAPLACIAN_TEST_V0.2\cache\pip"
$env:JOVIN_HARNESS_RUNTIME = "wsl:JovinHarness"
$env:JOVIN_HARNESS_TIMEOUT_S = "30"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
Set-Content -Path "D:\LLM_LAPLACIAN_TEST_V0.2\runs\queue.pid" -Value $PID -Encoding ascii
Add-Content -Path "D:\LLM_LAPLACIAN_TEST_V0.2\runs\queue.log" -Encoding ascii -Value "PROGRESS queue_resume shell_pid=$PID"
cmd /c ".\.venv\Scripts\python.exe -u scripts\run_queue.py >> runs\queue.log 2>&1"
exit $LASTEXITCODE
