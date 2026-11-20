$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$environmentPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) {
    $pythonCandidate = Get-Command python,python3 -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -notlike '*WindowsApps*' } | Select-Object -First 1
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if ($pythonCandidate) { $bootstrapPython = $pythonCandidate.Source }
    elseif (Test-Path -LiteralPath $bundledPython) { $bootstrapPython = $bundledPython }
    else { throw 'Python 3.12+ is required. Install Python and run this script again.' }
    & $bootstrapPython -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the virtual environment.' }
}
& $environmentPython -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
