param(
    [switch]$RunPyInstaller,
    [string]$SpecPath = "src\translator.spec",
    [string]$DistDir = "src\dist",
    [string]$ExecutablePattern = "Glossarion v*.exe",
    [string]$PackageName = "Shirochi.Glossarion",
    [string]$Publisher = "CN=Shirochi",
    [string]$PublisherDisplayName = "Shirochi",
    [string]$DisplayName = "Glossarion",
    [string]$Description = "Glossarion translation toolkit",
    [string]$PackageVersion = "8.6.5.0",
    [ValidateSet("x64", "x86", "arm", "arm64", "neutral")]
    [string]$Architecture = "x64",
    [string]$MinVersion = "10.0.17763.0",
    [string]$MaxVersionTested = "10.0.26100.0",
    [string]$IconPath = "src\Halgakos.png",
    [string]$OutputDir = "dist\msix",
    [string]$StageRoot = "build\msix",
    [string]$BackgroundColor = "transparent",
    [switch]$Sign,
    [string]$CertificatePath = "",
    [string]$CertificatePassword = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

function Get-WindowsSdkTool([string]$ToolName) {
    $sdkRoot = "${env:ProgramFiles(x86)}\Windows Kits\10\bin"
    if (-not (Test-Path -LiteralPath $sdkRoot)) {
        throw "Windows SDK bin directory was not found: $sdkRoot"
    }

    $candidates = @()
    foreach ($dir in Get-ChildItem -LiteralPath $sdkRoot -Directory) {
        $candidate = Join-Path $dir.FullName "x64\$ToolName"
        if (Test-Path -LiteralPath $candidate) {
            $candidates += [pscustomobject]@{
                Path = $candidate
                Version = $dir.Name
            }
        }
    }

    foreach ($archDir in @("x64", "x86")) {
        $candidate = Join-Path $sdkRoot "$archDir\$ToolName"
        if (Test-Path -LiteralPath $candidate) {
            $candidates += [pscustomobject]@{
                Path = $candidate
                Version = "0.0.0.0"
            }
        }
    }

    if (-not $candidates) {
        throw "$ToolName was not found. Install the Windows SDK or add $ToolName to PATH."
    }

    return ($candidates | Sort-Object Version -Descending | Select-Object -First 1).Path
}

function Convert-ToPackageVersion([string]$VersionValue) {
    $parts = $VersionValue.Split(".")
    if ($parts.Count -lt 3 -or $parts.Count -gt 4) {
        throw "PackageVersion must be 3 or 4 numeric parts, for example 8.6.5.0."
    }
    while ($parts.Count -lt 4) {
        $parts += "0"
    }
    foreach ($part in $parts) {
        $number = 0
        if (-not [int]::TryParse($part, [ref]$number) -or $number -lt 0 -or $number -gt 65535) {
            throw "PackageVersion contains an invalid part: $part. Each part must be 0-65535."
        }
    }
    return ($parts -join ".")
}

function Escape-Xml([string]$Value) {
    return [System.Security.SecurityElement]::Escape($Value)
}

function New-LogoAsset([string]$SourcePath, [string]$TargetPath, [int]$Width, [int]$Height) {
    Add-Type -AssemblyName System.Drawing
    $source = [System.Drawing.Image]::FromFile($SourcePath)
    try {
        $bitmap = New-Object System.Drawing.Bitmap $Width, $Height
        try {
            $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
            try {
                $graphics.Clear([System.Drawing.Color]::Transparent)
                $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
                $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
                $scale = [Math]::Min($Width / $source.Width, $Height / $source.Height)
                $drawWidth = [int]($source.Width * $scale)
                $drawHeight = [int]($source.Height * $scale)
                $left = [int](($Width - $drawWidth) / 2)
                $top = [int](($Height - $drawHeight) / 2)
                $graphics.DrawImage($source, $left, $top, $drawWidth, $drawHeight)
            }
            finally {
                $graphics.Dispose()
            }
            $bitmap.Save($TargetPath, [System.Drawing.Imaging.ImageFormat]::Png)
        }
        finally {
            $bitmap.Dispose()
        }
    }
    finally {
        $source.Dispose()
    }
}

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$SpecFullPath = Resolve-RepoPath $SpecPath
$DistFullPath = Resolve-RepoPath $DistDir
$IconFullPath = Resolve-RepoPath $IconPath
$OutputFullPath = Resolve-RepoPath $OutputDir
$StageFullPath = Resolve-RepoPath $StageRoot
$PackageRoot = Join-Path $StageFullPath "package"
$AppRoot = Join-Path $PackageRoot "App"
$AssetsRoot = Join-Path $PackageRoot "Assets"
$PackageVersion = Convert-ToPackageVersion $PackageVersion

if ($RunPyInstaller) {
    if (-not (Test-Path -LiteralPath $SpecFullPath)) {
        throw "Spec file not found: $SpecFullPath"
    }
    Push-Location (Split-Path -Parent $SpecFullPath)
    try {
        & python -m PyInstaller (Split-Path -Leaf $SpecFullPath)
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $DistFullPath)) {
    throw "Dist directory not found: $DistFullPath. Run with -RunPyInstaller or build translator.spec first."
}

$sourceExe = Get-ChildItem -LiteralPath $DistFullPath -File -Filter $ExecutablePattern |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $sourceExe) {
    throw "No executable matching '$ExecutablePattern' found in $DistFullPath."
}

if (-not (Test-Path -LiteralPath $IconFullPath)) {
    throw "Icon source image not found: $IconFullPath"
}

$resolvedPackageRoot = [System.IO.Path]::GetFullPath($PackageRoot)
$resolvedStageRoot = [System.IO.Path]::GetFullPath($StageFullPath)
if (-not $resolvedPackageRoot.StartsWith($resolvedStageRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean a package root outside the staging directory: $resolvedPackageRoot"
}

if (Test-Path -LiteralPath $PackageRoot) {
    Remove-Item -LiteralPath $PackageRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $AppRoot, $AssetsRoot, $OutputFullPath -Force | Out-Null

$stagedExe = Join-Path $AppRoot "Glossarion.exe"
Copy-Item -LiteralPath $sourceExe.FullName -Destination $stagedExe -Force

New-LogoAsset -SourcePath $IconFullPath -TargetPath (Join-Path $AssetsRoot "Square44x44Logo.png") -Width 44 -Height 44
New-LogoAsset -SourcePath $IconFullPath -TargetPath (Join-Path $AssetsRoot "Square150x150Logo.png") -Width 150 -Height 150
New-LogoAsset -SourcePath $IconFullPath -TargetPath (Join-Path $AssetsRoot "StoreLogo.png") -Width 50 -Height 50

$manifest = @"
<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
  IgnorableNamespaces="uap rescap">
  <Identity
    Name="$(Escape-Xml $PackageName)"
    Publisher="$(Escape-Xml $Publisher)"
    Version="$(Escape-Xml $PackageVersion)"
    ProcessorArchitecture="$(Escape-Xml $Architecture)" />
  <Properties>
    <DisplayName>$(Escape-Xml $DisplayName)</DisplayName>
    <PublisherDisplayName>$(Escape-Xml $PublisherDisplayName)</PublisherDisplayName>
    <Logo>Assets\StoreLogo.png</Logo>
  </Properties>
  <Dependencies>
    <TargetDeviceFamily
      Name="Windows.Desktop"
      MinVersion="$(Escape-Xml $MinVersion)"
      MaxVersionTested="$(Escape-Xml $MaxVersionTested)" />
  </Dependencies>
  <Resources>
    <Resource Language="en-us" />
  </Resources>
  <Applications>
    <Application
      Id="App"
      Executable="App\Glossarion.exe"
      EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements
        DisplayName="$(Escape-Xml $DisplayName)"
        Description="$(Escape-Xml $Description)"
        BackgroundColor="$(Escape-Xml $BackgroundColor)"
        Square150x150Logo="Assets\Square150x150Logo.png"
        Square44x44Logo="Assets\Square44x44Logo.png" />
    </Application>
  </Applications>
  <Capabilities>
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
"@

$manifestPath = Join-Path $PackageRoot "AppxManifest.xml"
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($manifestPath, $manifest, $utf8NoBom)

$makeAppx = Get-WindowsSdkTool "makeappx.exe"
$packageFileName = "{0}_{1}_{2}.msix" -f ($PackageName -replace '[^A-Za-z0-9_.-]', ''), $PackageVersion, $Architecture
$packagePath = Join-Path $OutputFullPath $packageFileName

& $makeAppx pack /d $PackageRoot /p $packagePath /o
if ($LASTEXITCODE -ne 0) {
    throw "makeappx failed with exit code $LASTEXITCODE."
}

if ($Sign) {
    if (-not $CertificatePath) {
        throw "Use -CertificatePath when signing locally, or omit -Sign for an unsigned Store-upload package."
    }
    $certFullPath = Resolve-RepoPath $CertificatePath
    if (-not (Test-Path -LiteralPath $certFullPath)) {
        throw "Certificate file not found: $certFullPath"
    }

    $signTool = Get-WindowsSdkTool "signtool.exe"
    $signArgs = @("sign", "/fd", "SHA256", "/f", $certFullPath)
    if ($CertificatePassword) {
        $signArgs += @("/p", $CertificatePassword)
    }
    if ($TimestampUrl) {
        $signArgs += @("/tr", $TimestampUrl, "/td", "SHA256")
    }
    $signArgs += $packagePath
    & $signTool @signArgs
    if ($LASTEXITCODE -ne 0) {
        throw "signtool failed with exit code $LASTEXITCODE."
    }
}

Write-Host "MSIX package created:" -ForegroundColor Green
Write-Host "  $packagePath"
Write-Host ""
Write-Host "Staged executable:"
Write-Host "  $($sourceExe.FullName)"
if (-not $Sign) {
    Write-Host ""
    Write-Host "This package is unsigned. That is intended for Microsoft Store submission, where the Store re-signs MSIX packages after certification." -ForegroundColor Yellow
    Write-Host "For local double-click install testing, rerun with -Sign and a trusted test certificate."
}
