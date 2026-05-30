param(
    [string]$ApkPath = "$env:USERPROFILE\Downloads\Glossarion-Android-PySide-debug (3)\Glossarion-0.1-arm64-v8a-debug.apk",
    [int]$WaitSeconds = 15
)

$ErrorActionPreference = "Stop"

function Resolve-Adb {
    $fromPath = Get-Command adb -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }

    $sdkAdb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
    if (Test-Path -LiteralPath $sdkAdb) {
        return $sdkAdb
    }

    throw "adb was not found. Install Android SDK platform-tools or add adb.exe to PATH."
}

function Invoke-Adb {
    & $script:AdbPath @args
}

$script:AdbPath = Resolve-Adb
$apk = Resolve-Path -LiteralPath $ApkPath
$packageName = "org.glossarion.glossarion"
$outputDir = Join-Path (Split-Path -Parent $apk) ("runtime-logs-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

Write-Host "Using adb: $script:AdbPath"
Write-Host "Using APK: $apk"
Write-Host "Writing logs to: $outputDir"

$deviceLines = Invoke-Adb devices | Select-Object -Skip 1 | Where-Object { $_ -match "\tdevice\b" }
if (-not $deviceLines) {
    Invoke-Adb devices -l | Tee-Object -FilePath (Join-Path $outputDir "adb-devices.txt")
    throw "No authorized adb device/emulator is connected. Enable USB debugging and accept the device prompt."
}

Invoke-Adb devices -l | Tee-Object -FilePath (Join-Path $outputDir "adb-devices.txt")

$abiList = (Invoke-Adb shell getprop ro.product.cpu.abilist 2>$null) -join ""
$apkName = Split-Path -Leaf $apk
if ($apkName -match "arm64-v8a" -and $abiList -and $abiList -notmatch "arm64-v8a") {
    throw "Connected target ABI is '$abiList', but APK is arm64-v8a. Build an x86_64 APK for the local emulator or use an arm64 device."
}
if ($apkName -match "x86_64" -and $abiList -and $abiList -notmatch "x86_64") {
    throw "Connected target ABI is '$abiList', but APK is x86_64."
}

Invoke-Adb install -r -d "$apk" 2>&1 | Tee-Object -FilePath (Join-Path $outputDir "adb-install.txt")
Invoke-Adb logcat -c
Invoke-Adb shell monkey -p $packageName -c android.intent.category.LAUNCHER 1 2>&1 |
    Tee-Object -FilePath (Join-Path $outputDir "adb-launch.txt")

Start-Sleep -Seconds $WaitSeconds

$logcatPath = Join-Path $outputDir "logcat.txt"
Invoke-Adb logcat -d -v threadtime 2>&1 | Out-File -FilePath $logcatPath -Encoding utf8

$filteredPath = Join-Path $outputDir "logcat-filtered.txt"
Select-String -Path $logcatPath -Pattern (
    "AndroidRuntime",
    "FATAL EXCEPTION",
    "Traceback",
    "Glossarion",
    "Python",
    "PySide",
    "Qt",
    "libpython",
    "crash",
    "exception"
) | ForEach-Object { $_.Line } | Out-File -FilePath $filteredPath -Encoding utf8

$startupCrashPath = Join-Path $outputDir "glossarion_pyside_startup_crash.txt"
try {
    Invoke-Adb shell run-as $packageName sh -c 'find . -name glossarion_pyside_startup_crash.log -type f -print -exec cat {} \;' 2>&1 |
        Out-File -FilePath $startupCrashPath -Encoding utf8
} catch {
    "run-as startup crash pull failed: $_" | Out-File -FilePath $startupCrashPath -Encoding utf8
}

Write-Host "Captured:"
Write-Host "  $logcatPath"
Write-Host "  $filteredPath"
Write-Host "  $startupCrashPath"
