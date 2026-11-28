param([switch]$IncludeLegacy,[switch]$Overwrite)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$environmentPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) { & (Join-Path $PSScriptRoot 'setup.ps1') }
& $environmentPython -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Algorithm checks failed.' }
if ($Overwrite -and -not $IncludeLegacy) { throw '-Overwrite applies only to -IncludeLegacy. Preserve the frozen new protocol and use a separate copy for a full rerun.' }
if ($IncludeLegacy) {
    $experimentArguments = @('-m','experiments.run','--suite','all','--device','cpu','--threads','2')
    if ($Overwrite) { $experimentArguments += '--overwrite' }
    & $environmentPython @experimentArguments
    if ($LASTEXITCODE -ne 0) { throw 'Legacy experiments failed.' }
    & $environmentPython -m experiments.summarize
    if ($LASTEXITCODE -ne 0) { throw 'Legacy aggregation failed.' }
}
# New experiments run on the Linux CUDA server via scripts/run_h20.sh.
# This Windows entry point only rebuilds artifacts from downloaded records.
& $environmentPython plot.py
if ($LASTEXITCODE -ne 0) { throw 'Revised plotting and aggregation failed.' }
& (Join-Path $PSScriptRoot 'build_report.ps1')
& $environmentPython (Join-Path $PSScriptRoot 'verify_artifacts.py')
if ($LASTEXITCODE -ne 0) { throw 'Artifact verification failed.' }
