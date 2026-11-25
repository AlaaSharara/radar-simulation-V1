$ErrorActionPreference = "Stop"
Write-Host "=== Radar simulation environment setup ===" -ForegroundColor Cyan

$launcher = Get-Command py -ErrorAction SilentlyContinue
if ($launcher) {
    $pythonTag = $null
    foreach ($candidate in @("-3.12", "-3.11")) {
        & py $candidate --version 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonTag = $candidate; break }
    }
    if (-not $pythonTag) { throw "Python 3.11 or 3.12 not found. Install one and retry." }
    if (-not (Test-Path ".venv")) { & py $pythonTag -m venv .venv }
} else {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) { throw "Python not found. Install Python 3.11 or 3.12." }
    & python --version
    if ($LASTEXITCODE -ne 0) { throw "Python could not be started." }
    if (-not (Test-Path ".venv")) { & python -m venv .venv }
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
& .\.venv\Scripts\python.exe -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "Installing the project package failed." }

if (Test-Path "viz\web\package.json") {
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npm) { Write-Warning "Node.js/npm not found. Install Node.js 20 LTS to run the dashboard." }
    else {
        Push-Location "viz\web"
        try { npm install; if ($LASTEXITCODE -ne 0) { throw "npm install failed." } } finally { Pop-Location }
    }
}

Write-Host "Setup complete. Run .\run_backend.ps1 and .\run_frontend.ps1 in separate PowerShell windows." -ForegroundColor Green
