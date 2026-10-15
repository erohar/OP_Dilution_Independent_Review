@echo off
cd /d "%~dp0"
python build.py --install %*
exit /b %ERRORLEVEL%
