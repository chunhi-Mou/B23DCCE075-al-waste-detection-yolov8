@echo off
REM Chay ung dung demo Gradio o local. Mo http://127.0.0.1:7860
REM Demo can checkpoint best.pt cua baseline trong results\ hoac runs\baseline\
setlocal
cd /d "%~dp0.."
set PYTHONPATH=%CD%

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

%PY% -c "import gradio, ultralytics, torch" 2>nul
if errorlevel 1 (
    echo Cai dat thu vien tu requirements.txt ...
    %PY% -m pip install -r requirements.txt
)

%PY% -m demo.app --no-share

pause
