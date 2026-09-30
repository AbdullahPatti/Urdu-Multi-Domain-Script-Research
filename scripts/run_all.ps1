# Full evaluation matrix, unattended and resumable (finished cells are skipped).
# Launch detached from any terminal via Task Scheduler:  scripts\launch.cmd
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:HF_HUB_CACHE = "E:\hf_cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:TQDM_DISABLE = "1"
New-Item -ItemType Directory -Force logs | Out-Null

function Stage($name, $log, $pyargs) {
    "[{0}] START {1}" -f (Get-Date -Format s), $name | Add-Content logs\progress.log
    # cmd redirection keeps the log UTF-8 (PowerShell 5.1 *>> would write UTF-16)
    cmd /c ".venv312\Scripts\python.exe $pyargs >> $log 2>&1"
    "[{0}] END   {1} (exit {2})" -f (Get-Date -Format s), $name, $LASTEXITCODE | Add-Content logs\progress.log
}

Stage "fine-tuning (XLM-R, mBERT x 3 seeds)" "logs\finetune.log" "-m pipeline.run_finetune"
foreach ($m in @("Qwen2.5-7B", "Mistral-7B", "Llama-3.1-8B")) {
    Stage "LLM $m (3 exemplar seeds)" "logs\llm_$m.log" "-m pipeline.run_llm --models $m"
}
Stage "analysis" "logs\analyze.log" "-m pipeline.analyze"
"[{0}] ALL DONE" -f (Get-Date -Format s) | Add-Content logs\progress.log
