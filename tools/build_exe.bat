@echo off
REM Builds dist\OfflineOCR\OfflineOCR.exe from the installed .venv (run install.bat first).
REM Needs PyInstaller in .venv:  .venv\Scripts\python -m pip install --no-index --find-links build_tools pyinstaller
setlocal
cd /d "%~dp0.."
set "VPY=%CD%\.venv\Scripts\python.exe"
"%VPY%" -m PyInstaller --noconfirm --clean --distpath dist --workpath build tools\OfflineOCR.spec || exit /b 1
echo Copying models and data folders next to the exe ...
robocopy models dist\OfflineOCR\models /E /NFL /NDL /NJH /NJS >nul
for %%D in (user datasets output) do if not exist "dist\OfflineOCR\%%D" mkdir "dist\OfflineOCR\%%D"
if exist user\corrections.json copy /Y user\corrections.json dist\OfflineOCR\user\ >nul
echo.
echo Done: dist\OfflineOCR\OfflineOCR.exe
endlocal
