param([string]$Database)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Install the Poetry environment (.venv) first.' }
$directory = Join-Path $root '.tools\local-stack'
New-Item -ItemType Directory -Path $directory -Force | Out-Null
$ready = Join-Path $directory 'ready.json'
$id = [guid]::NewGuid().ToString('N')
if (Test-Path -LiteralPath $ready) {
    $running = Get-Content -LiteralPath $ready -Raw | ConvertFrom-Json
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($running.pid)" -ErrorAction SilentlyContinue
    if ($process.CommandLine -like "*scripts.local_stack*--run-id $($running.run_id)*") {
        Write-Output 'NewsSummarizer already running. Streamlit http://127.0.0.1:8501 | Swagger http://127.0.0.1:8000/docs'
        return
    }
}
$arguments = "-X utf8 -m scripts.local_stack --run-id $id"
if ($Database) {
    $path = [IO.Path]::GetFullPath($Database)
    if ($path.Contains('"')) { throw 'Invalid database path.' }
    $arguments += " --database `"$path`""
}
$errors = Join-Path $directory "$id.stderr.log"
$launch = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $directory "$id.stdout.log") -RedirectStandardError $errors -PassThru
for ($i = 0; $i -lt 180; $i++) {
    if (Test-Path -LiteralPath $ready) {
        $running = Get-Content -LiteralPath $ready -Raw | ConvertFrom-Json
        if ($running.run_id -eq $id) {
            Write-Output 'Streamlit http://127.0.0.1:8501 | Swagger http://127.0.0.1:8000/docs'
            return
        }
    }
    $launch.Refresh()
    if ($launch.HasExited) { break }
    Start-Sleep -Milliseconds 500
}
if (Test-Path -LiteralPath $errors) { Get-Content -LiteralPath $errors -Tail 12 -Encoding UTF8 }
if (-not $launch.HasExited) { New-Item -ItemType File -Path (Join-Path $directory "$id.stop") -Force | Out-Null }
throw "Cannot start NewsSummarizer. See $directory."
