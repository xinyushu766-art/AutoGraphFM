param(
    [string]$Destination = "data/srni_car/Sales.csv",
    [string]$OutputDir = "outputs/srni_real",
    [switch]$Quick,
    [switch]$SkipNeural,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$urls = @(
    "https://figshare.com/ndownloader/files/42438726?private_link=255ef697e8b7e3115087",
    "https://ndownloader.figshare.com/files/42438726?private_link=255ef697e8b7e3115087"
)

$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$destinationDirectory = Split-Path -Parent $destinationPath
New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
if ((Test-Path $destinationPath) -and -not $Overwrite) {
    throw "Destination already exists: $destinationPath. Use -Overwrite to replace it."
}

$downloaded = $false
foreach ($url in $urls) {
    try {
        Write-Host "Trying $url"
        Invoke-WebRequest -Uri $url -OutFile $destinationPath -UseBasicParsing
        $downloaded = $true
        break
    }
    catch {
        Write-Warning $_.Exception.Message
        Remove-Item -ErrorAction SilentlyContinue $destinationPath
    }
}

if (-not $downloaded) {
    throw "Automatic Figshare download failed. Download Sales CSV from https://srni-car.github.io/ and save it to $destinationPath."
}

$validationDir = Join-Path $OutputDir "00_srni_inspection"
python scripts/validate_srni_car_file.py `
    --sales-csv $destinationPath `
    --output-dir $validationDir `
    --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$arguments = @(
    "scripts/run_srni_car_pipeline.py",
    "--sales-csv", $destinationPath,
    "--output-dir", $OutputDir,
    "--train-end", "2020-12",
    "--validation-end", "2021-12",
    "--test-end", "2022-12",
    "--min-history-months", "36",
    "--horizons", "1", "3", "6",
    "--seeds", "42", "43", "44",
    "--overwrite"
)
if ($Quick) { $arguments += "--quick" }
if ($SkipNeural) { $arguments += "--skip-neural" }
python @arguments
exit $LASTEXITCODE
