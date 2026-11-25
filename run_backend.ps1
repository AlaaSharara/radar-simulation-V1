$ErrorActionPreference = "Stop"
$PythonExe = if (Test-Path ".venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$env:PYTHONPATH = "src"
& $PythonExe scripts\validate_config.py
if ($LASTEXITCODE -ne 0) { throw "Radar configuration validation failed." }
& $PythonExe -m radarsim.main --config radar_config.yaml --scene scene.json --mode preview
