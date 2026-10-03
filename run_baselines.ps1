$env:HF_DATASETS_CACHE='D:\hfd'; $env:PYTHONIOENCODING='utf-8'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
Set-Location D:\arabic8b
$tasks = 'arabic_leaderboard_arabic_mmlu_light,arabic_exams_light,arabic_leaderboard_acva_light'
foreach ($m in @(@('D:/arabic8b/models/Qwen3-8B','qwen3-8b-base'), @('D:/arabic8b/models/ALLaM-7B','allam-7b'))) {
  "=== $($m[1]) start $(Get-Date -Format s)"
  .\.venv\Scripts\python.exe run_eval.py $m[0] $m[1] $tasks 2>&1 | Select-String -NotMatch 'RemoteException|it/s\]|Warn|warn'
  "=== $($m[1]) end $(Get-Date -Format s)"
}
