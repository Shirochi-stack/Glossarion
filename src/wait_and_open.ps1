# Wait for Gradio server to be ready and then open browser
param(
    [string]$url = "http://127.0.0.1:7860",
    [int]$maxWaitSeconds = 60
)

Write-Host "Waiting for server to be ready at $url..." -ForegroundColor Cyan

$startTime = Get-Date
$ready = $false

while (-not $ready -and ((Get-Date) - $startTime).TotalSeconds -lt $maxWaitSeconds) {
    try {
        $response = Invoke-WebRequest -Uri $url -Method Head -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $ready = $true
            Write-Host "Server is ready!" -ForegroundColor Green
        }
    }
    catch {
        # Server not ready yet, wait a bit
        Start-Sleep -Milliseconds 500
    }
}

if ($ready) {
    Write-Host "Opening browser..." -ForegroundColor Green
    Start-Process $url
} else {
    Write-Host "Timeout waiting for server. Please open $url manually." -ForegroundColor Yellow
}