$ErrorActionPreference = "Stop"

$root = "C:\deepfake_eval\hidf_eval"
$raw = Join-Path $root "raw"
$downloads = Join-Path $raw "downloads"
$python = "C:\Users\weissach\Desktop\Y3\Y3-project\zoom-frame-server\venv\Scripts\python.exe"
$script = "C:\Users\weissach\Desktop\Y3\Y3-project\automation\windows\prepare_hidf_eval.py"
$sevenZip = "C:\Program Files\7-Zip\7z.exe"

$downloadMap = @{
    "Fake-vid.zip" = "https://zenodo.org/records/16140829/files/Fake-vid.zip?download=1"
    "Real-vid.zip" = "https://zenodo.org/records/16140829/files/Real-vid.zip?download=1"
    "metadata.csv" = "https://zenodo.org/records/16140829/files/metadata.csv?download=1"
}

New-Item -ItemType Directory -Force -Path $downloads | Out-Null
New-Item -ItemType Directory -Force -Path $raw | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root "supplementary\real") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root "supplementary\fake") | Out-Null

foreach ($name in $downloadMap.Keys) {
    $target = Join-Path $downloads $name
    if (-not (Test-Path $target)) {
        curl.exe -L $downloadMap[$name] -o $target
    }
}

$extractMap = @(
    @{ Archive = (Join-Path $downloads "Fake-vid.zip"); OutDir = (Join-Path $raw "Fake-vid") },
    @{ Archive = (Join-Path $downloads "Real-vid.zip"); OutDir = (Join-Path $raw "Real-vid") }
)

foreach ($item in $extractMap) {
    if (-not (Test-Path $item.OutDir)) {
        New-Item -ItemType Directory -Force -Path $item.OutDir | Out-Null
        & $sevenZip x $item.Archive "-o$($item.OutDir)" -y | Out-Null
    }
}

$metadataTarget = Join-Path $raw "metadata.csv"
if (-not (Test-Path $metadataTarget)) {
    Copy-Item -Force (Join-Path $downloads "metadata.csv") $metadataTarget
}

& $python $script --root $root
