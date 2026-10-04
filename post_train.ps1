# Runs unattended after train.py: waits for the training process to exit, then evaluates,
# samples, merges and quantizes. Every step logs to post_train.log; a failed step stops the chain.
$env:PYTHONIOENCODING='utf-8'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'; $env:HF_DATASETS_CACHE='D:\hfd'
Set-Location D:\arabic8b
$py = '.\.venv\Scripts\python.exe'
function Step($name, [scriptblock]$body) {
  "=== $name start $(Get-Date -Format s)"
  & $body 2>&1 | Where-Object { "$_" -notmatch 'it/s\]|s/it\]|RemoteException|SentencePiece|Falling back|triton' }
  if ($LASTEXITCODE) { "=== $name FAILED exit $LASTEXITCODE $(Get-Date -Format s)"; exit 1 }
  "=== $name ok $(Get-Date -Format s)"
}

while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like '*train.py*') {
  Start-Sleep 60
}
if (-not (Test-Path 'D:\arabic8b\runs\allam-ar-sft\final\adapter_config.json')) {
  "=== training ended without a final adapter $(Get-Date -Format s)"; exit 1
}
Step 'eval-tuned' { & $py run_eval.py D:/arabic8b/models/ALLaM-7B allam-7b-sft arabic_leaderboard_arabic_mmlu_light,arabic_exams_light,arabic_leaderboard_acva_light 0 D:/arabic8b/runs/allam-ar-sft/final }
Step 'samples' { & $py gen_samples.py }
Step 'merge-quantize' { & $py merge_quantize.py }
"=== ALL DONE $(Get-Date -Format s)"
