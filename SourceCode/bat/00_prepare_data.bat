@echo off
REM Giai doan chuan bi du lieu: audit -> tach split dong bang -> export 3 thu muc -> bieu do phan bo.
REM Chi chay khi dung lai du lieu tu Dataset\ tho. Neu da co export\ thi bo qua file nay.
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

echo [1/4] Audit dataset tho (Dataset\)
%PY% scripts\01_audit_dataset.py || goto :err

echo [2/4] Tao split dong bang (splits\)
%PY% scripts\02_make_splits.py --config configs\benchmark.yaml --seeds configs\seeds.yaml || goto :err

echo [3/4] Export dataset 3 thu muc (export\)
%PY% scripts\06_export_dataset.py --config configs\benchmark.yaml --out export || goto :err

echo [4/4] Bieu do phan bo lop (reports\charts\)
%PY% scripts\07_distribution_charts.py --data export\data.yaml --out reports\charts || goto :err

echo.
echo Xong. Du lieu san sang tai export\
pause
exit /b 0

:err
echo.
echo THAT BAI. Xem thong bao loi o tren.
pause
exit /b 1
