@echo off
rem Start scripts\run_all.ps1 through Task Scheduler so it is independent of any terminal
rem or editor session. Safe to re-run: finished result cells are skipped.
rem The trigger date is far in the future on purpose: the task only runs when started
rem here with /Run (a same-day trigger once fired by itself in the middle of a data rebuild).
schtasks /Query /TN "UrduRobustnessRun" /FO LIST 2>nul | findstr /C:"Running" >nul && (
    echo Task is already running - not starting a second copy.
    exit /b 1
)
schtasks /Create /TN "UrduRobustnessRun" /SC ONCE /SD 01/01/2099 /ST 00:00 /F /TR "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File \"%~dp0run_all.ps1\"" >nul
schtasks /Run /TN "UrduRobustnessRun"
