$ErrorActionPreference = "Stop"
if (-not (Test-Path "viz\web\package.json")) { throw "Frontend not created yet. Complete Antigravity task T0/T6 first." }
if (-not (Test-Path "viz\web\node_modules")) { Push-Location "viz\web"; try { npm install } finally { Pop-Location } }
Push-Location "viz\web"
try { npm run dev -- --host 127.0.0.1 } finally { Pop-Location }
