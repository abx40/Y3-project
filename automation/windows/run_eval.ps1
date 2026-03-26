[CmdletBinding()]
param(
    [string]$EvalRoot,
    [string]$Models = "all",
    [string]$Splits = "",
    [string]$Kinds = "",
    [string]$Ordinals = "",
    [switch]$SmokeTest,
    [switch]$ResumeFailed,
    [double]$DurationLimitSeconds,
    [double]$SmokeDurationSeconds = 45,
    [double]$BufferSeconds = 15,
    [string]$Browser,
    [string]$OBSHost = "127.0.0.1",
    [int]$OBSPort = 4455,
    [string]$OBSPassword = $env:OBS_WEBSOCKET_PASSWORD,
    [string]$OBSScene = "Y3 Eval",
    [string]$OBSInput = "Y3 Eval Media",
    [string]$BrowserCameraLabel = "OBS Virtual Camera",
    [string]$BotImage = "vsdk-1.11.2-on-ubuntu",
    [switch]$SkipBuildBotImage,
    [switch]$DryRun,
    [switch]$SkipBot
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$localNodeCandidates = @((Join-Path $repoRoot "tools\node"))
$localNodeVersionDirs = Get-ChildItem -Path (Join-Path $repoRoot "tools") -Directory -Filter "node-v*-win-x64" -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    ForEach-Object { $_.FullName }
if ($localNodeVersionDirs) {
    $localNodeCandidates += $localNodeVersionDirs
}
$pythonCandidates = @(
    (Join-Path $repoRoot "zoom-frame-server\venv\Scripts\python.exe"),
    "python"
)

foreach ($candidate in $localNodeCandidates) {
    if (Test-Path (Join-Path $candidate "node.exe")) {
        $env:PATH = "$candidate;$env:PATH"
        break
    }
}

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -eq "python") {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) {
            $pythonExe = $cmd.Source
            break
        }
    } elseif (Test-Path $candidate) {
        $pythonExe = (Resolve-Path $candidate).Path
        break
    }
}

if (-not $pythonExe) {
    throw "Could not find a Python interpreter. Expected zoom-frame-server\venv\Scripts\python.exe or python on PATH."
}

$orchestrator = Join-Path $scriptDir "orchestrator.py"
$argsList = @($orchestrator, "--models", $Models, "--buffer-seconds", "$BufferSeconds", "--smoke-duration-seconds", "$SmokeDurationSeconds", "--obs-host", $OBSHost, "--obs-port", "$OBSPort", "--obs-scene", $OBSScene, "--obs-input", $OBSInput, "--browser-camera-label", $BrowserCameraLabel, "--bot-image", $BotImage)

if ($EvalRoot) { $argsList += @("--eval-root", $EvalRoot) }
if ($Splits) { $argsList += @("--splits", $Splits) }
if ($Kinds) { $argsList += @("--kinds", $Kinds) }
if ($Ordinals) { $argsList += @("--ordinals", $Ordinals) }
if ($Browser) { $argsList += @("--browser", $Browser) }
if ($OBSPassword) { $argsList += @("--obs-password", $OBSPassword) }
if ($SmokeTest) { $argsList += "--smoke-test" }
if ($ResumeFailed) { $argsList += "--resume-failed" }
if ($PSBoundParameters.ContainsKey("DurationLimitSeconds")) { $argsList += @("--duration-limit-seconds", "$DurationLimitSeconds") }
if ($SkipBuildBotImage) { $argsList += "--skip-build-bot-image" }
if ($DryRun) { $argsList += "--dry-run" }
if ($SkipBot) { $argsList += "--skip-bot" }

Write-Host "[run_eval] repo_root=$repoRoot"
Write-Host "[run_eval] python=$pythonExe"
Write-Host "[run_eval] orchestrator=$orchestrator"

& $pythonExe @argsList
exit $LASTEXITCODE
