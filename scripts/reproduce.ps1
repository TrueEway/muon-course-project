param([switch]$Overwrite)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$environmentPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) { & (Join-Path $PSScriptRoot 'setup.ps1') }
& $environmentPython -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Algorithm checks failed.' }
$experimentArguments = @('-m','experiments.run','--suite','all','--device','cpu','--threads','2')
if ($Overwrite) { $experimentArguments += '--overwrite' }
& $environmentPython @experimentArguments
if ($LASTEXITCODE -ne 0) { throw 'Experiments failed.' }
& $environmentPython -m experiments.summarize
if ($LASTEXITCODE -ne 0) { throw 'Result aggregation failed.' }
& (Join-Path $PSScriptRoot 'build_report.ps1')
& $environmentPython (Join-Path $PSScriptRoot 'verify_artifacts.py')
if ($LASTEXITCODE -ne 0) { throw 'Artifact verification failed.' }
