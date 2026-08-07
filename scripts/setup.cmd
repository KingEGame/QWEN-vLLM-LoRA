@echo off
setlocal EnableExtensions
set "WSL_UTF8=1"

rem Resolve repo root = parent of this scripts\ directory
set "SCRIPT_DIR=%~dp0"
rem Strip trailing backslash for path math
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "REPO_ROOT=%%~fI"

where wsl >nul 2>&1
if errorlevel 1 (
  echo ERROR: wsl.exe not found. Install WSL2 first:
  echo   wsl --install -d Ubuntu
  echo Then reboot and re-run scripts\setup.cmd
  exit /b 1
)

rem Prefer exact name Ubuntu; else first Ubuntu* from `wsl -l -q`
set "DISTRO="
for /f "usebackq delims=" %%D in (`wsl -l -q`) do (
  if /I "%%D"=="Ubuntu" set "DISTRO=Ubuntu"
)
if not defined DISTRO (
  for /f "usebackq delims=" %%D in (`wsl -l -q`) do (
    echo %%D | findstr /I /B "Ubuntu" >nul && (
      if not defined DISTRO set "DISTRO=%%D"
    )
  )
)
if not defined DISTRO (
  echo ERROR: No Ubuntu WSL distro found. Install one with:
  echo   wsl --install -d Ubuntu
  exit /b 1
)

rem Convert Windows path to WSL /mnt/<drive>/...
set "WSL_REPO="
set "WSLPATH_OUT=%TEMP%\qwen-setup-wslpath-%RANDOM%.txt"
wsl -d %DISTRO% -e wslpath -a "%REPO_ROOT%" >"%WSLPATH_OUT%" 2>&1
set "WSLPATH_RC=%ERRORLEVEL%"
set /p WSL_REPO=<"%WSLPATH_OUT%"
del "%WSLPATH_OUT%" >nul 2>&1
if not "%WSLPATH_RC%"=="0" (
  echo ERROR: wslpath failed ^(exit %WSLPATH_RC%^) converting repo path: %REPO_ROOT%
  echo %WSL_REPO%
  exit /b 1
)
if not defined WSL_REPO (
  echo ERROR: Failed to convert repo path to a WSL path: %REPO_ROOT%
  exit /b 1
)
if not "%WSL_REPO:~0,1%"=="/" (
  echo ERROR: Unexpected wslpath output ^(expected a path starting with /^): %WSL_REPO%
  exit /b 1
)

echo Using WSL distro: %DISTRO%
echo Repo in WSL: %WSL_REPO%
echo Running scripts/setup.sh ...

wsl -d %DISTRO% -e bash -lc "cd \"%WSL_REPO%\" && bash scripts/setup.sh"
exit /b %ERRORLEVEL%
