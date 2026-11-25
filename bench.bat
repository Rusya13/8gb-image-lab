@echo off
rem Measures one model on this PC:  bench.bat qwen-turbo   (or moss, klein, qwen)
rem Run run.bat once first so the environment exists. Rows go to results\results.csv.
setlocal
cd /d "%~dp0"
if "%~1"=="" ( echo Usage: bench.bat qwen-turbo ^| moss ^| klein ^| qwen & exit /b 1 )
".venv\Scripts\python.exe" bench.py --model %*
