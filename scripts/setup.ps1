#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$env:WSL_UTF8 = "1"

$ScriptDir = $PSScriptRoot
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
    Write-Host @"
ERROR: wsl.exe not found. Install WSL2 first:
  wsl --install -d Ubuntu
Then reboot and re-run scripts\setup.ps1
"@
    exit 1
}

$distro = Get-UbuntuDistroName
if (-not $distro) {
    Write-Host @"
ERROR: No Ubuntu WSL distro found. Install one with:
  wsl --install -d Ubuntu
"@
    exit 1
}

$wslRepo = (wsl -d $distro -e wslpath -a $RepoRoot | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: wslpath failed (exit $LASTEXITCODE) converting repo path: $RepoRoot"
    exit 1
}
if (-not $wslRepo) {
    Write-Host "ERROR: Failed to convert repo path to a WSL path: $RepoRoot"
    exit 1
}
if (-not $wslRepo.StartsWith("/")) {
    Write-Host "ERROR: Unexpected wslpath output (expected a path starting with /): $wslRepo"
    exit 1
}

Write-Host "Using WSL distro: $distro"
Write-Host "Repo in WSL: $wslRepo"
Write-Host "Running scripts/setup.sh ..."

wsl -d $distro -e bash -lc "cd `"$wslRepo`" && bash scripts/setup.sh"
exit $LASTEXITCODE
