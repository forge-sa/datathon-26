@echo off
rem ============================================================================
rem  Cash-discipline video audit: one-click launcher.
rem  Double-click this file. It asks for the settings (Enter = keep the value in
rem  brackets), remembers them in .env, then starts Ollama + the web service and
rem  shows the log here (also written to logs\server_*.log).
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Cash audit service

rem ---- defaults, then the values saved last time --------------------------------
set "API_KEY="
set "PORT=8000"
set "BIND_HOST=127.0.0.1"
set "OFFLINE=Y"
set "RESTART_OLLAMA=N"
set "OPEN_BROWSER=Y"
set "TUNNEL_NAME="
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do set "%%a=%%b"
)
if "!API_KEY!"=="" (
    for /f %%k in ('powershell -NoProfile -Command "[guid]::NewGuid().ToString('N')"') do set "API_KEY=%%k"
)

echo.
echo  =====================================================================
echo   Cash audit service - settings   (Enter = keep the value in [ ])
echo  =====================================================================
echo.
call :ask API_KEY        "API key for the jury (X-API-Key)"
call :ask PORT           "Port"
call :ask BIND_HOST      "Listen address (127.0.0.1 = only this PC / tunnel)"
call :ask OFFLINE        "Offline mode, no internet for models (Y/N)"
call :ask RESTART_OLLAMA "Restart Ollama with recommended GPU settings (Y/N)"
call :ask OPEN_BROWSER   "Open the web page when ready (Y/N)"
call :ask TUNNEL_NAME    "Cloudflare tunnel to start (empty = skip, already running)"

rem ---- remember the answers ---------------------------------------------------------
(
    echo # saved by start.bat - local settings, never commit this file
    echo API_KEY=!API_KEY!
    echo PORT=!PORT!
    echo BIND_HOST=!BIND_HOST!
    echo OFFLINE=!OFFLINE!
    echo RESTART_OLLAMA=!RESTART_OLLAMA!
    echo OPEN_BROWSER=!OPEN_BROWSER!
    echo TUNNEL_NAME=!TUNNEL_NAME!
) > ".env"

echo.
echo  Starting... (close this window or press Ctrl+C to stop the service)
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" -Port !PORT! -BindHost "!BIND_HOST!" -ApiKey "!API_KEY!" -Offline "!OFFLINE!" -RestartOllama "!RESTART_OLLAMA!" -OpenBrowser "!OPEN_BROWSER!" -TunnelName "!TUNNEL_NAME!"
echo.
echo  The service has stopped. See the logs folder for the full log.
pause
exit /b

:ask
set "CUR=!%~1!"
set "IN="
set /p "IN=  %~2 [!CUR!]: "
if not "!IN!"=="" set "%~1=!IN!"
exit /b
