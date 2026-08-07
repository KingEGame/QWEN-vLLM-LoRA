#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

function Get-UbuntuDistroName {
    $names = @(wsl -l -q 2>$null | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ })
    if (-not $names) {
        return $null
    }
    $exact = $names | Where-Object { $_ -eq "Ubuntu" } | Select-Object -First 1
    if ($exact) {
        return $exact
    }
    return ($names | Where-Object { $_ -like "Ubuntu*" } | Select-Object -First 1)
}

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Error @"
wsl.exe not found. Install WSL2 first:
  wsl --install -d Ubuntu
Then reboot and re-run scripts\setup.ps1
"@
    exit 1
}

$distro = Get-UbuntuDistroName
if (-not $distro) {
    Write-Error @"
No Ubuntu WSL distro found. Install one with:
  wsl --install -d Ubuntu
"@
    exit 1
}

$wslRepo = (wsl -d $distro -e wslpath -a $RepoRoot).Trim()
if (-not $wslRepo) {
    Write-Error "Failed to convert repo path to a WSL path: $RepoRoot"
    exit 1
}

Write-Host "Using WSL distro: $distro"
Write-Host "Repo in WSL: $wslRepo"
Write-Host "Running scripts/setup.sh ..."

wsl -d $distro -e bash -lc "cd `"$wslRepo`" && bash scripts/setup.sh"
exit $LASTEXITCODE
