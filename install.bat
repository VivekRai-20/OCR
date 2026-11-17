@echo off
REM ==========================================================================
REM  Offline OCR - offline installer.  No internet is used.
REM  1. finds Python 3.12 (or installs it from installers\ for this user)
REM  2. creates .venv inside this folder
REM  3. installs every package from wheels\ (pip --no-index)
REM  4. checks that all models and packages load
REM  5. puts an "Offline OCR" shortcut on the desktop
REM ==========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "PYEXE="

for /f "delims=" %%P in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if not defined PYEXE (
  for /f "delims=" %%P in ('python -c "import sys;assert sys.version_info[:2]==(3,12);print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
)
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE (
  for %%I in (installers\python-3.12*-amd64.exe) do set "PYINST=%%I"
)
if not defined PYEXE (
  if not defined PYINST (
    echo Python 3.12 64-bit was not found and installers\ has no Python installer.
    exit /b 1
  )
  echo Installing Python 3.12 for the current user from %PYINST% ...
  "%PYINST%" /passive InstallAllUsers=0 PrependPath=0 Include_launcher=1 Include_test=0 Include_doc=0
  set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
if not exist "%PYEXE%" (
  echo Python not found at "%PYEXE%".
  exit /b 1
)
echo Using Python: %PYEXE%

if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment .venv ...
  "%PYEXE%" -m venv .venv || exit /b 1
)
set "VPY=%~dp0.venv\Scripts\python.exe"
"%VPY%" -m pip install --no-index --find-links wheels --upgrade pip setuptools wheel
echo Installing packages from wheels\ (offline) ...
"%VPY%" -m pip install --no-index --find-links wheels -r requirements.txt
if errorlevel 1 (
  echo.
  echo Package installation failed. See the messages above.
  exit /b 1
)
REM NVIDIA GPU present -> CUDA build of torch, so the AI page reader runs ~10x faster
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  if exist "wheels\gpu\torch-*.whl" (
    echo NVIDIA GPU found - installing the CUDA build of torch from wheels\gpu ...
    for %%W in (wheels\gpu\torch-*.whl wheels\gpu\torchvision-*.whl) do "%VPY%" -m pip install --no-index --no-deps --force-reinstall "%%W"
  )
)
"%VPY%" tools\check_install.py || exit /b 1

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Offline OCR.lnk');" ^
  "$s.TargetPath='%~dp0.venv\Scripts\pythonw.exe'; $s.Arguments='\"%~dp0app.py\"'; $s.WorkingDirectory='%~dp0'; $s.Save()" >nul 2>nul

echo.
echo Installation complete. Start the app with run.bat or the "Offline OCR" desktop shortcut.
endlocal
