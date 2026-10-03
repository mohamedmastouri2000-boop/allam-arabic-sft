$env:PYTHONIOENCODING='utf-8'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'; $env:BS='1'
Set-Location D:\arabic8b
"=== start $(Get-Date -Format s)"
.\.venv\Scripts\python.exe train.py
"=== end $(Get-Date -Format s) exit $LASTEXITCODE"
