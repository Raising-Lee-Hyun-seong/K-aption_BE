# K-aption benchmark v2: 새 영상 / 스레드 6 / 강의별 용어집 / GPU 를 단계별로 측정한다.
# 결과: bench\out2\<단계>\ (results.csv, 응답 JSON, 해당 단계 backend 로그)
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Bench = $PSScriptRoot
$Root = Split-Path -Parent $Bench
Set-Location $Root
$Out2 = Join-Path $Bench "out2"
New-Item -ItemType Directory -Force $Out2 | Out-Null
$Work = Join-Path $env:TEMP "kaption_bench"
New-Item -ItemType Directory -Force $Work | Out-Null
$Base = @("-f", "compose.yaml")

function Log($m) { $l = "$(Get-Date -Format s) $m"; Write-Host $l; Add-Content (Join-Path $Out2 "status.txt") $l }

function Up($files, $label) {
    Log "compose up ($label)"
    docker compose @files up -d *>&1 | Out-File -Append (Join-Path $Out2 "compose_$label.txt")
    if ($LASTEXITCODE -ne 0) { Log "compose failed ($label) exit=$LASTEXITCODE"; return $false }
    for ($i = 0; $i -lt 120; $i++) {
        try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5; if ($h.status -eq "ok") { Log "healthy ($label)"; return $true } } catch { }
        Start-Sleep 5
    }
    Log "NOT healthy ($label)"; return $false
}

function Get-Clip($row) {
    $p = $row.Split('|'); $name = $p[0].Trim(); $media = Join-Path $Work "$name.mp4"
    if (-not (Test-Path $media)) { Log "download $name"; curl.exe -s -L --max-time 900 -o $media $p[1].Trim() }
    return @{ name = $name; media = $media; vtt = $(if ($p.Count -gt 2) { $p[2].Trim() } else { "" }) }
}

function Run-Phase($phase, $rows, $warmup) {
    $dir = Join-Path $Out2 $phase; New-Item -ItemType Directory -Force $dir | Out-Null
    $since = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $csv = Join-Path $dir "results.csv"; "name,run,http_status,client_seconds,bytes" | Set-Content $csv
    $first = $true
    foreach ($row in $rows) {
        $c = Get-Clip $row
        if ($c.vtt) { curl.exe -s -L --max-time 120 -o (Join-Path $dir "$($c.name).vtt") $c.vtt }
        if (-not (Test-Path $c.media) -or (Get-Item $c.media).Length -lt 10000) { Log "download failed $($c.name)"; continue }
        $runs = if ($first -and $warmup) { @("warmup", "1") } else { @("1") }; $first = $false
        foreach ($run in $runs) {
            Log "[$phase] request $($c.name) run=$run"
            $sw = [Diagnostics.Stopwatch]::StartNew()
            $code = curl.exe -s --max-time 3600 -o (Join-Path $dir "$($c.name).$run.json") -w "%{http_code}" -F "file=@$($c.media)" "http://127.0.0.1:8000/api/v1/transcriptions?language=en"
            $sec = [math]::Round($sw.Elapsed.TotalSeconds, 3)
            "$($c.name),$run,$code,$sec,$((Get-Item $c.media).Length)" | Add-Content $csv
            Log "[$phase] done $($c.name) run=$run status=$code seconds=$sec"
        }
    }
    docker compose logs backend --no-color --since $since *>&1 | Out-File (Join-Path $dir "backend_logs.txt")
    docker stats --no-stream *>&1 | Out-File (Join-Path $dir "docker_stats.txt")
}

$old = Get-Content (Join-Path $Bench "clips.txt") | Where-Object { $_ -match '\|' -and -not $_.StartsWith('#') }
$new = Get-Content (Join-Path $Bench "clips_new.txt") | Where-Object { $_ -match '\|' -and -not $_.StartsWith('#') }
function Pick($names) { $old | Where-Object { $names -contains $_.Split('|')[0] } }

Log "start v2"
# A. 새 영상 3개 (기존 설정: 번역 8스레드, 물리 용어집)
$env:KAPTION_CPU_THREADS = "8"; $env:KAPTION_TRANSLATOR_CPUS = "8"
if (Up $Base "A") { Run-Phase "A_new_clips" $new $true }

# C1. 생물 용어집 (번역 8스레드)
$env:KAPTION_GLOSSARY_FILE = "bio_terms.json"
if (Up ($Base + @("-f", "bench/compose.glossary.yaml")) "C1") { Run-Phase "C1_bio_glossary" (Pick @("bio_lac_operon", "bio_pcr")) $false }
# C2. CS 용어집
$env:KAPTION_GLOSSARY_FILE = "cs_terms.json"
if (Up ($Base + @("-f", "bench/compose.glossary.yaml")) "C2") { Run-Phase "C2_cs_glossary" (Pick @("cs_for_loops_strings")) $false }

# C3. 자동 생성 용어집: 1차 결과의 STT 원문으로 로컬 Qwen이 강의별 용어집을 만든 뒤 같은 클립을 다시 번역
$gfiles = $Base + @("-f", "bench/compose.glossary.yaml")
$env:KAPTION_GLOSSARY_FILE = "bio_terms.json"
if (Up $gfiles "C3gen") {
    foreach ($clip in @("bio_lac_operon", "bio_pcr", "cs_for_loops_strings")) {
        Log "auto glossary $clip"
        $lines = docker compose @gfiles exec -T -e PYTHONIOENCODING=utf-8 backend python /scripts/build_glossary.py --input "/bench/out/$clip.1.json" 2>> (Join-Path $Out2 "auto_glossary_err.txt")
        $text = ($lines -join "`n")
        try { $n = ($text | ConvertFrom-Json).Count } catch { $n = 0 }
        if ($n -gt 8) {
            [IO.File]::WriteAllText((Join-Path $Bench "glossary\auto_$clip.json"), $text, (New-Object Text.UTF8Encoding($false)))
            Log "auto glossary $clip terms=$($n - 8)"
        } else { Log "auto glossary $clip failed" }
    }
    foreach ($clip in @("bio_lac_operon", "bio_pcr", "cs_for_loops_strings")) {
        if (Test-Path (Join-Path $Bench "glossary\auto_$clip.json")) {
            $env:KAPTION_GLOSSARY_FILE = "auto_$clip.json"
            if (Up $gfiles "C3_$clip") { Run-Phase "C3_auto_$clip" (Pick @($clip)) $false }
        }
    }
}

# B. 번역 6스레드 (용어집 원래대로)
$env:KAPTION_CPU_THREADS = "6"; $env:KAPTION_TRANSLATOR_CPUS = "6"
if (Up $Base "B") { Run-Phase "B_threads6" (Pick @("phys_rocket", "bio_lac_operon")) $true }

# D. GPU (RTX 3060 Ti, 같은 GGUF 모델)
$env:KAPTION_CPU_THREADS = "8"; $env:KAPTION_TRANSLATOR_CPUS = "8"
Log "pull cuda image"
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda *>&1 | Out-File (Join-Path $Out2 "pull_gpu.txt")
if (Up ($Base + @("-f", "bench/compose.gpu.yaml")) "D") {
    docker compose -f compose.yaml -f bench/compose.gpu.yaml logs translator --no-color *>&1 | Select-Object -Last 80 | Out-File (Join-Path $Out2 "gpu_translator_log.txt")
    Run-Phase "D_gpu" (Pick @("phys_rocket", "bio_lac_operon", "bio_pcr")) $true
} else {
    docker compose -f compose.yaml -f bench/compose.gpu.yaml logs translator --no-color *>&1 | Select-Object -Last 80 | Out-File (Join-Path $Out2 "gpu_translator_log.txt")
}

# E. 외부 API 번역 (KAPTION_API_KEY 환경변수가 있을 때만). 새 코드로 backend를 다시 빌드한다.
if ($env:KAPTION_API_KEY) {
    Log "build backend for API mode"
    docker compose build backend *>&1 | Out-File (Join-Path $Out2 "build_api.txt")
    if (Up ($Base + @("-f", "bench/compose.api.yaml")) "E") {
        Run-Phase "E_api_$($env:KAPTION_API_MODEL)" (Pick @("phys_rocket", "bio_lac_operon", "cs_for_loops_strings")) $false
    }
} else { Log "skip E (no KAPTION_API_KEY)" }

# 원래 CPU 구성으로 되돌리기
Up $Base "restore" | Out-Null
Log "ALL DONE v2"
