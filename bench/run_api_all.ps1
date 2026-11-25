# 여러 외부 모델을 차례로 비교한다. 키($env:KAPTION_API_KEY)와 주소($env:KAPTION_API_URL)를 넣은 창에서 실행.
#   powershell -ExecutionPolicy Bypass -File bench\run_api_all.ps1
# 모델마다 bench\out3\<모델>\ 에 결과와 Qwen 비교표가 생기고, 끝나면 로컬 Qwen 구성으로 되돌린다.
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Bench = $PSScriptRoot
Set-Location (Split-Path -Parent $Bench)
if (-not $env:KAPTION_API_KEY) { Write-Host "KAPTION_API_KEY가 비어 있습니다."; exit 1 }
if (-not $env:KAPTION_API_URL) { $env:KAPTION_API_URL = "https://openrouter.ai/api" }
$log = Join-Path $Bench "out3\run_all_status.txt"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Log($m) { $l = "$(Get-Date -Format s) $m"; Write-Host $l; Add-Content $log $l }

# 비교할 모델 후보. OpenRouter에 없는 이름은 건너뛰고 비슷한 이름을 기록한다.
$candidates = @(
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "google/gemini-3.5-flash-lite",
    "deepseek/deepseek-v4-flash",
    "anthropic/claude-haiku-5.5",
    "qwen/qwen3-235b-a22b-2507"
)
try { $ids = (Invoke-RestMethod -Uri "https://openrouter.ai/api/v1/models" -TimeoutSec 30).data.id } catch { $ids = @(); Log "모델 목록을 못 받아 이름 확인 없이 진행" }

Log "start ($($candidates.Count)개 후보)"
foreach ($m in $candidates) {
    if ($ids.Count -gt 0 -and $ids -notcontains $m) {
        $vendor = $m.Split('/')[0]; $key = ($m.Split('/')[1] -split '-')[0]
        $near = ($ids | Where-Object { $_ -like "$vendor/*$key*" } | Select-Object -First 8) -join ", "
        Log "skip $m (OpenRouter에 없음). 비슷한 이름: $near"
        continue
    }
    $env:KAPTION_API_MODEL = $m
    Log "run $m"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $Bench "run_api.ps1") *>&1 | Tee-Object -Append -FilePath $log | Out-Host
    Log "done $m"
}
Log "로컬 Qwen 구성으로 되돌림"
docker compose up -d backend *>&1 | Out-Null
Log "ALL DONE"
