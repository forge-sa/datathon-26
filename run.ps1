# Start the service. Easiest: double-click start.bat (asks for these settings and calls this script).
# Direct use:  powershell -ExecutionPolicy Bypass -File run.ps1 -ApiKey <key> [-Port 8000] [-BindHost 127.0.0.1]
#              [-Offline Y] [-RestartOllama N] [-OpenBrowser N] [-TunnelName <cloudflare tunnel>]
param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [string]$ApiKey = "forte-dev-key",
    [string]$Offline = "N",          # Y: Hugging Face offline (RT-DETR from the local cache, no internet)
    [string]$RestartOllama = "N",    # Y: stop a running Ollama and start it with the settings below
    [string]$OpenBrowser = "N",      # Y: open the web page once the service answers
    [string]$TunnelName = ""         # name of a Cloudflare tunnel to start in its own window (empty: skip)
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Model = "qwen3-vl:8b-instruct"
function Yes($v) { return "$v".Trim().ToUpper().StartsWith("Y") }
function Step($msg) { Write-Host "[start] $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "[start] ERROR: $msg" -ForegroundColor Red; exit 1 }

# ---- checks ---------------------------------------------------------------------------
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Fail "Python environment .venv not found. Set it up first (README section 1)."
}
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Fail "Ollama is not installed or not on PATH (https://ollama.com/download)."
}
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    $proc = (Get-Process -Id $busy[0].OwningProcess -ErrorAction SilentlyContinue).ProcessName
    Fail "port $Port is already in use (process: $proc). Stop the old server or choose another port."
}

# ---- Ollama ---------------------------------------------------------------------------
# These settings only apply to an Ollama server started here (the tray app ignores them).
$env:OLLAMA_KEEP_ALIVE = "-1"
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_NUM_PARALLEL = "1"    # 2 pushes part of the 8B model to CPU on 12 GB
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"

function OllamaUp { try { Invoke-RestMethod http://127.0.0.1:11434/api/tags -TimeoutSec 2 | Out-Null; return $true } catch { return $false } }

if ((Yes $RestartOllama) -and (OllamaUp)) {
    Step "stopping the running Ollama..."
    Get-Process "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
}
if (-not (OllamaUp)) {
    Step "starting Ollama (keep-alive, flash attention, 1 parallel slot, q8 KV cache)..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not (OllamaUp); $i++) { Start-Sleep -Seconds 1 }
    if (-not (OllamaUp)) { Fail "Ollama did not start." }
} else {
    Step "Ollama is already running (reusing it)."
}

$tags = (Invoke-RestMethod http://127.0.0.1:11434/api/tags).models.name
if ($tags -notcontains $Model) {
    Step "model $Model not found - downloading (about 6 GB, needs internet)..."
    & ollama pull $Model
    if ($LASTEXITCODE -ne 0) { Fail "ollama pull $Model failed." }
}

Step "loading $Model into GPU memory (first time can take a minute)..."
try {
    $body = '{"model":"' + $Model + '","prompt":"ok","stream":false,"keep_alive":-1,"options":{"num_predict":1,"num_ctx":6144}}'
    Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body $body -ContentType "application/json" -TimeoutSec 180 | Out-Null
    $ps = (Invoke-RestMethod http://127.0.0.1:11434/api/ps).models | Where-Object { $_.name -eq $Model }
    if ($ps -and $ps.size_vram -lt $ps.size) {
        Write-Host "[start] WARNING: part of the model is in system RAM (slow). Close GPU-heavy apps and restart with 'Restart Ollama = Y'." -ForegroundColor Yellow
    }
} catch { Write-Host "[start] WARNING: model warm-up failed: $_" -ForegroundColor Yellow }
try {
    $gpu = & nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>$null
    if ($gpu) { Step "GPU: $gpu" }
} catch {}

# ---- Cloudflare tunnel (optional) ------------------------------------------------------
New-Item -ItemType Directory -Force logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ("$TunnelName".Trim()) {
    if (Get-Command cloudflared -ErrorAction SilentlyContinue) {
        Step "starting Cloudflare tunnel '$TunnelName' in its own window (log: logs\tunnel_$stamp.log)..."
        Start-Process cloudflared -ArgumentList "tunnel", "--logfile", "logs\tunnel_$stamp.log", "run", $TunnelName
    } else {
        Write-Host "[start] WARNING: cloudflared not found - tunnel not started." -ForegroundColor Yellow
    }
}

# ---- web service --------------------------------------------------------------------------
$env:PYTHONUTF8 = "1"
$env:API_KEY = $ApiKey
$env:LOG_FILE = "logs\server_$stamp.log"
if (Yes $Offline) { $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1" }

$url = "http://${BindHost}:$Port/"
if ($BindHost -eq "0.0.0.0") { $url = "http://127.0.0.1:$Port/" }
if (Yes $OpenBrowser) {
    # wait in the background until the service answers, then open the page
    $wait = "for (`$i=0; `$i -lt 120; `$i++) { try { Invoke-RestMethod '${url}health' -TimeoutSec 2 | Out-Null; Start-Process '$url'; break } catch { Start-Sleep 1 } }"
    Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile", "-Command", $wait
}

Write-Host ""
Write-Host "  Web UI : $url" -ForegroundColor Green
Write-Host "  API    : ${url}api/v1/analyze   (header X-API-Key: $ApiKey)" -ForegroundColor Green
Write-Host "  Health : ${url}health" -ForegroundColor Green
Write-Host "  Log    : $PSScriptRoot\$($env:LOG_FILE)" -ForegroundColor Green
Write-Host "  Offline: $(if (Yes $Offline) {'yes'} else {'no'})    Stop: Ctrl+C or close this window"
Write-Host ""
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host $BindHost --port $Port
