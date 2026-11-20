$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$temporaryRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$buildRoot = Join-Path $temporaryRoot ('muon-latex-' + [guid]::NewGuid().ToString('N'))
if ($buildRoot -match '[^\x00-\x7F]') { throw 'TeX Live needs an ASCII temporary path on this Windows installation.' }
New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'report') -Destination $buildRoot -Recurse
New-Item -ItemType Directory -Path (Join-Path $buildRoot 'output') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'output\figures'),(Join-Path $projectRoot 'output\tables') -Destination (Join-Path $buildRoot 'output') -Recurse
New-Item -ItemType Directory -Path (Join-Path $buildRoot 'output\pdf') -Force | Out-Null
Push-Location -LiteralPath (Join-Path $buildRoot 'report')
try {
    New-Item -ItemType Directory -Path (Join-Path $projectRoot 'output\pdf') -Force | Out-Null
    & latexmk -xelatex -interaction=nonstopmode -halt-on-error '-outdir=../output/pdf' main.tex
    $compilationExit = $LASTEXITCODE
    Get-ChildItem -LiteralPath (Join-Path $buildRoot 'output\pdf') -File | Copy-Item -Destination (Join-Path $projectRoot 'output\pdf') -Force
    if ($compilationExit -ne 0) { throw 'LaTeX compilation failed.' }
} finally {
    Pop-Location
    $resolvedBuild = [System.IO.Path]::GetFullPath($buildRoot)
    if (-not $resolvedBuild.StartsWith($temporaryRoot, [System.StringComparison]::OrdinalIgnoreCase) -or (Split-Path -Leaf $resolvedBuild) -notmatch '^muon-latex-[0-9a-f]{32}$') {
        throw 'Temporary build directory did not pass the cleanup path check.'
    }
    Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
}
