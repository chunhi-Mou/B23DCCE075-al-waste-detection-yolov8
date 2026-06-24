@echo off
REM Mo hinh doi chung (Oracle): huan luyen YOLOv8n tren 100%% du lieu, seed 13.
setlocal
cd /d "%~dp0.."
set PYTHONPATH=%CD%

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

%PY% -c "import ultralytics, torch, yaml" 2>nul
if errorlevel 1 (
    echo Cai dat thu vien tu requirements.txt ...
    %PY% -m pip install -r requirements.txt || goto :err
)

%PY% scripts\03_train_baseline.py --config configs\benchmark.yaml --data export\data.yaml --frac 1.0 || goto :err

echo.
echo Xong baseline. Ket qua o runs\baseline\ va reports\baseline\
pause
exit /b 0

:err
echo.
echo THAT BAI. Xem thong bao loi o tren.
pause
exit /b 1
