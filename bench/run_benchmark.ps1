# K-aption benchmark runner (Windows PowerShell)
# 1) Docker compose build/up  2) wait for health  3) wait for bench\clips.txt
# 4) download clips, warm-up once, upload each clip and time it  5) save logs to bench\out
$ErrorActionPreference = "Continue"
$Bench = $PSScriptRoot
$Root = Split-Path -Parent $Bench
Set-Location $Root
$Out = Join-Path $Bench "out"
New-Item -ItemType Directory -Force $Out | Out-Null
$Work = Join-Path $env:TEMP "kaption_bench"
New-Item -ItemType Directory -Force $Work | Out-Null

function Log($m) {
    $line = "$(Get-Date -Format s) $m"
    Write-Host $line
    Add-Content -Path (Join-Path $Out "status.txt") -Value $line
}

Log "start"
for ($i = 0; $i -lt 90; $i++) {
    docker info *> $null
    if ($LASTEXITCODE -eq 0) { break }
    Log "waiting for Docker engine ($i)"
    Start-Sleep 10
}
docker version 2>&1 | Out-File (Join-Path $Out "docker_version.txt")
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors |
    Format-List | Out-File (Join-Path $Out "env_cpu.txt")
docker info --format "{{.NCPU}} cpus {{.MemTotal}} bytes" 2>&1 | Out-File (Join-Path $Out "env_docker.txt")

# same translator setting as the 2026-11-01 Mac test (8 CPU / 8 threads)
$env:KAPTION_CPU_THREADS = "8"
$env:KAPTION_TRANSLATOR_CPUS = "8"
Log "compose build start"
$sw = [Diagnostics.Stopwatch]::StartNew()
docker compose up -d --build *>&1 | Out-File (Join-Path $Out "compose_build.txt")
Log "compose exit=$LASTEXITCODE seconds=$([int]$sw.Elapsed.TotalSeconds)"

$healthy = $false
for ($i = 0; $i -lt 360; $i++) {
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
        $h | ConvertTo-Json | Out-File (Join-Path $Out "health.json")
        if ($h.status -eq "ok" -or $h.model_loaded -or $h.translator_loaded) { $healthy = $true; break }
    } catch { }
    if ($i % 6 -eq 0) { Log "waiting for health ($i)" }
    Start-Sleep 10
}
Log "healthy=$healthy"
docker compose ps *>&1 | Out-File (Join-Path $Out "compose_ps.txt")

$Clips = Join-Path $Bench "clips.txt"
for ($i = 0; $i -lt 540 -and -not (Test-Path $Clips); $i++) {
    if ($i % 30 -eq 0) { Log "waiting for clips.txt" }
    Start-Sleep 10
}
if (-not (Test-Path $Clips)) { Log "no clips.txt, stop"; exit 1 }

$csv = Join-Path $Out "results.csv"
"name,run,http_status,client_seconds,bytes" | Set-Content $csv
$rows = Get-Content $Clips | Where-Object { $_ -match '\|' -and -not $_.StartsWith('#') }
$first = $true
foreach ($row in $rows) {
    $p = $row.Split('|')
    $name = $p[0].Trim(); $url = $p[1].Trim(); $vtt = if ($p.Count -gt 2) { $p[2].Trim() } else { "" }
    $media = Join-Path $Work "$name.mp4"
    if (-not (Test-Path $media)) {
        Log "download $name"
        curl.exe -s -L --max-time 900 -o $media $url
    }
    if ($vtt) { curl.exe -s -L --max-time 120 -o (Join-Path $Out "$name.vtt") $vtt }
    if (-not (Test-Path $media) -or (Get-Item $media).Length -lt 10000) { Log "download failed $name"; continue }
    $runs = if ($first) { @("warmup", "1") } else { @("1") }
    $first = $false
    foreach ($run in $runs) {
        Log "request $name run=$run"
        $resp = Join-Path $Out "$name.$run.json"
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $code = curl.exe -s --max-time 3600 -o $resp -w "%{http_code}" -F "file=@$media" "http://127.0.0.1:8000/api/v1/transcriptions?language=en"
        $sec = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        "$name,$run,$code,$sec,$((Get-Item $media).Length)" | Add-Content $csv
        Log "done $name run=$run status=$code seconds=$sec"
    }
}
docker compose logs backend --no-color *>&1 | Out-File (Join-Path $Out "backend_logs.txt")
docker stats --no-stream *>&1 | Out-File (Join-Path $Out "docker_stats.txt")
Log "ALL DONE"
