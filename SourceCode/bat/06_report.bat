@echo off
REM Tong hop ket qua AL: bieu do, bang AUBC, kiem dinh t-test -> reports\al\
REM Chay sau khi da co du ket qua cua baseline va 4 chien luoc.
setlocal
cd /d "%~dp0.."
set PYTHONPATH=%CD%

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

%PY% -c "import ultralytics, torch, yaml, scipy" 2>nul
if errorlevel 1 (
    echo Cai dat thu vien tu requirements.txt ...
    %PY% -m pip install -r requirements.txt || goto :err
)

%PY% scripts\11_al_report.py --config configs\benchmark.yaml --data export\data.yaml --out reports\al || goto :err

echo.
echo Xong bao cao. Ket qua o reports\al\
pause
exit /b 0

:err
echo.
echo THAT BAI. Xem thong bao loi o tren.
pause
exit /b 1
