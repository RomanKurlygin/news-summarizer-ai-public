$ErrorActionPreference = 'Stop'
$directory = Join-Path (Split-Path -Parent $PSScriptRoot) '.tools\local-stack'
$ready = Join-Path $directory 'ready.json'
if (-not (Test-Path -LiteralPath $ready)) { Write-Output 'NewsSummarizer local runner is not active.'; return }
$running = Get-Content -LiteralPath $ready -Raw | ConvertFrom-Json
if ($running.run_id -notmatch '^[0-9a-f]{32}$') { throw 'Invalid local runner state.' }
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $($running.pid)" -ErrorAction SilentlyContinue
if ($process.CommandLine -notlike "*scripts.local_stack*--run-id $($running.run_id)*") {
    Write-Output 'NewsSummarizer local runner is not active.'
    return
}
New-Item -ItemType File -Path (Join-Path $directory "$($running.run_id).stop") -Force | Out-Null
for ($i = 0; $i -lt 140; $i++) {
    if (-not (Test-Path -LiteralPath $ready)) { Write-Output 'NewsSummarizer services stopped.'; return }
    Start-Sleep -Milliseconds 500
}
throw 'Local runner did not finish. Check .tools/local-stack logs.'
