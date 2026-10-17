# Start the service:  powershell -ExecutionPolicy Bypass -File run.ps1
# Options:            -Port 8000 -BindHost 127.0.0.1 -ApiKey forte-dev-key
param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [string]$ApiKey = "forte-dev-key"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:API_KEY = $ApiKey

# Ollama: reuse a running server, otherwise start one with the recommended settings.
# (Settings only apply to a server started here; quit the Ollama tray app first to use them.)
$running = $true
try { Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 2 | Out-Null } catch { $running = $false }
if (-not $running) {
    $env:OLLAMA_KEEP_ALIVE = "-1"
    $env:OLLAMA_FLASH_ATTENTION = "1"
    $env:OLLAMA_NUM_PARALLEL = "1"   # 2 pushes part of the 8B model to CPU on 12 GB
    $env:OLLAMA_KV_CACHE_TYPE = "q8_0"
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# Warm up the model so the first video doesn't pay the load time.
try {
    $body = '{"model":"qwen3-vl:8b-instruct","prompt":"ok","stream":false,"keep_alive":-1,"options":{"num_predict":1,"num_ctx":6144}}'
    Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120 | Out-Null
} catch { Write-Warning "Ollama warm-up failed: $_" }

Write-Host "UI:  http://${BindHost}:$Port/"
Write-Host "API: http://${BindHost}:$Port/api/v1/analyze   (X-API-Key: $ApiKey)"
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host $BindHost --port $Port
