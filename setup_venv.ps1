# Creates a fresh venv and installs all dependencies with CUDA-enabled PyTorch.
# Usage: .\setup.ps1
# Optional: .\setup.ps1 -VenvDir data_generator_venv

param(
    [string]$VenvDir = "data_generator_venv"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment: $VenvDir"
python -m venv $VenvDir

$pip = ".\$VenvDir\Scripts\pip.exe"
$python = ".\$VenvDir\Scripts\python.exe"

Write-Host "Upgrading pip..."
& $python -m pip install --upgrade pip --quiet

Write-Host "Installing PyTorch with CUDA 12.8 support..."
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing remaining dependencies..."
& $pip install -r requirements.txt

Write-Host ""
Write-Host "Verifying GPU availability..."
& $python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
