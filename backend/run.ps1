# Reality Debugger backend - one-step start for Windows PowerShell.
# Creates the virtual environment on first run, installs dependencies, starts uvicorn.
# If script execution is blocked:  Set-ExecutionPolicy -Scope Process RemoteSigned
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment (venv)..."
    & $python -m venv venv
}
& .\venv\Scripts\Activate.ps1
pip install --disable-pip-version-check -q -r requirements.txt
$port = if ($env:PORT) { $env:PORT } else { "8000" }
$hostAddr = if ($env:HOST) { $env:HOST } else { "0.0.0.0" }
Write-Host "Starting backend on http://localhost:$port (API docs: /api/docs)"
uvicorn app.main:app --host $hostAddr --port $port
