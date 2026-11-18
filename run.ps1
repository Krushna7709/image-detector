$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $root "src"

$cliArgs = $args
if ($cliArgs.Count -eq 0) {
    $cliArgs = @("serve")
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython -m aiforensics @cliArgs
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 -m aiforensics @cliArgs
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python -m aiforensics @cliArgs
} else {
    Write-Error "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/"
    exit 1
}
