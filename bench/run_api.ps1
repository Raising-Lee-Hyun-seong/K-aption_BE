# 외부 API 번역 비교: backend를 API 모드로 켠 뒤 같은 클립을 올려 bench\out3\<모델>\ 에 저장하고 Qwen 결과와 비교한다.
# 사용: $env:KAPTION_API_KEY / KAPTION_API_URL / KAPTION_API_MODEL 설정 후
#       powershell -ExecutionPolicy Bypass -File bench\run_api.ps1
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Bench = $PSScriptRoot
$Root = Split-Path -Parent $Bench
Set-Location $Root
if (-not $env:KAPTION_API_KEY) { Write-Host "KAPTION_API_KEY가 비어 있습니다. 키를 먼저 넣으세요."; exit 1 }
$model = if ($env:KAPTION_API_MODEL) { $env:KAPTION_API_MODEL } else { "gpt-5-mini" }
$dir = Join-Path $Bench ("out3\" + ($model -replace '[\\/:]', '_'))
New-Item -ItemType Directory -Force $dir | Out-Null
$Work = Join-Path $env:TEMP "kaption_bench"
$since = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

Write-Host "backend를 API 모드($model)로 켭니다"
docker compose -f compose.yaml -f bench/compose.api.yaml up -d --build backend
for ($i = 0; $i -lt 60; $i++) {
    try { if ((Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5).status -eq "ok") { break } } catch { }
    Start-Sleep 5
}

$csv = Join-Path $dir "results.csv"; "name,http_status,client_seconds" | Set-Content $csv
foreach ($clip in @("phys_rocket", "bio_lac_operon", "cs_for_loops_strings")) {
    $media = Join-Path $Work "$clip.mp4"
    if (-not (Test-Path $media)) { Write-Host "$clip 영상이 없습니다($media). 건너뜁니다."; continue }
    Write-Host "업로드 $clip ..."
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $code = curl.exe -s --max-time 3600 -o (Join-Path $dir "$clip.1.json") -w "%{http_code}" -F "file=@$media" "http://127.0.0.1:8000/api/v1/transcriptions?language=en"
    $sec = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    "$clip,$code,$sec" | Add-Content $csv
    Write-Host "  $clip status=$code ${sec}초"
}
docker compose logs backend --no-color --since $since *>&1 | Out-File (Join-Path $dir "backend_logs.txt") -Encoding utf8
python bench/compare_api.py --model-dir $dir
Write-Host "끝. 비교표는 $dir\compare_*.md 에 있습니다. 원래 Qwen으로 돌아가려면: docker compose up -d backend"
