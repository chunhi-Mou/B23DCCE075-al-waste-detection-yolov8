@echo off
REM S3 PPAL - chay 3 hat giong (13, 42, 1337).
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

%PY% scripts\10_run_al.py --config configs\benchmark.yaml --data export\data.yaml --strategy ppal || goto :err

echo.
echo Xong S3 PPAL. Ket qua o runs\al\ va state\al\
pause
exit /b 0

:err
echo.
echo THAT BAI. Xem thong bao loi o tren.
pause
exit /b 1
