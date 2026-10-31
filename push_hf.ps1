param(
    [string]$Token = ""
)

Set-Location $PSScriptRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  The Primer v2 - Hugging Face Hub Model Yukleyici" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

& "$PSScriptRoot\.venv\Scripts\hf.exe" auth whoami 2>$null
if ($LASTEXITCODE -ne 0) {
    if ($Token -eq "") {
        Write-Host ""
        Write-Host "1. https://huggingface.co/settings/tokens adresinden 'Write' yetkili token'inizi kopyalayin." -ForegroundColor Yellow
        Write-Host "2. Asagiya sag tiklayip (veya Ctrl+V ile) yapistirin ve Enter'a basin:" -ForegroundColor Yellow
        Write-Host ""
        $Token = (Read-Host "Hugging Face Write Token (hf_...)").Trim()
    }
    if ($Token -eq "") {
        Write-Host "Hata: Token bos birakildi." -ForegroundColor Red
        Read-Host "Cikmak icin Enter'a basin"
        exit 1
    }
    & "$PSScriptRoot\.venv\Scripts\hf.exe" auth login --token $Token
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Hata: Giris basarisiz oldu. Lutfen Write yetkili bir token girdiginizden emin olun." -ForegroundColor Red
        Read-Host "Cikmak icin Enter'a basin"
        exit 1
    }
}

Write-Host ""
Write-Host "Model Hugging Face Hub'a yukleniyor ve commit SHA sabitleniyor..." -ForegroundColor Green
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\hf_cli.py" push-model
if ($LASTEXITCODE -ne 0) {
    Read-Host "Hata olustu. Cikmak icin Enter'a basin"
    exit 1
}

Write-Host ""
Write-Host "Islem basariyla tamamlandi!" -ForegroundColor Green
Start-Sleep -Seconds 3
