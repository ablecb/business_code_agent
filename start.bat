@echo off
rem ============================================================
rem  Business Code Agent 一键启动 (后端 + 桌面版 + 网页版)
rem  后端: 系统 Python D:\Python\Python3.10.4 (依赖已装好)
rem  桌面: desktop\node_modules 已就绪 (含 electron 二进制)
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set PY=D:\Python\Python3.10.4\python.exe

echo [1/3] 启动后端 http://127.0.0.1:8000 ...
if not exist "%PY%" (
  echo   !! 未找到 %PY% ，请修改本脚本顶部的 PY 路径
  pause & exit /b 1
)

rem 若后端已在运行则跳过
powershell -NoProfile -Command "try { $null = Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/health' -TimeoutSec 3; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 (
  echo   !! 后端已在运行，跳过启动
  goto backend_ok
)

start "bca-backend" /min "%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
echo  等待后端就绪 ...
:wait_loop
powershell -NoProfile -Command "try { $null = Invoke-RestMethod 'http://127.0.0.1:8000/api/v1/health' -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 goto backend_ok
timeout /t 2 /nobreak >nul
goto wait_loop

:backend_ok
echo  后端就绪: http://127.0.0.1:8000

echo [2/3] 检查桌面端 ...
if not exist "desktop\node_modules\electron\dist\electron.exe" (
  echo   !! 桌面端 electron 缺失，仅启动网页版: http://127.0.0.1:8000
  goto done
)

echo [3/3] 启动桌面版 ...
cd /d "desktop"
start "" node_modules\.bin\electron.cmd .

:done
cd /d "%~dp0"
echo.
echo 完成。网页版: http://127.0.0.1:8000
echo 桌面版窗口已弹出（若没弹，直接双击 desktop\node_modules\.bin\electron.cmd）
echo 关闭本窗口不会关闭后端。
pause
