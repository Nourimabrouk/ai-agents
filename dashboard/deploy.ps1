# Enterprise Dashboard Deployment Script
# PowerShell script for Windows deployment of the Business Intelligence Dashboard

param(
    [string]$Environment = "development",
    [string]$Port = "8501",
    [switch]$Install,
    [switch]$Update,
    [switch]$Start,
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Status,
    [switch]$Clean,
    [switch]$Help
)

# Dashboard configuration
$DashboardName = "Enterprise Document Processing Dashboard"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RequirementsFile = "$ProjectRoot\..\requirements.txt"
$DashboardRequirements = "$ProjectRoot\requirements-dashboard.txt"
$MainDashboardFile = "$ProjectRoot\main_dashboard.py"
$LogFile = "$ProjectRoot\logs\dashboard.log"
$PidFile = "$ProjectRoot\dashboard.pid"

# Colors for output
$Red = "`e[31m"
$Green = "`e[32m"
$Yellow = "`e[33m"
$Blue = "`e[34m"
$Magenta = "`e[35m"
$Cyan = "`e[36m"
$Reset = "`e[0m"

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = $Reset
    )
    Write-Host "${Color}${Message}${Reset}"
}

function Show-Help {
    Write-ColoredOutput "🚀 Enterprise Dashboard Deployment Script" $Cyan
    Write-ColoredOutput "=" * 50 $Blue
    Write-Host ""
    Write-ColoredOutput "USAGE:" $Yellow
    Write-Host "  .\deploy.ps1 [OPTIONS]"
    Write-Host ""
    Write-ColoredOutput "OPTIONS:" $Yellow
    Write-Host "  -Install      Install all dependencies and setup environment"
    Write-Host "  -Update       Update dependencies to latest versions"
    Write-Host "  -Start        Start the dashboard server"
    Write-Host "  -Stop         Stop the dashboard server"
    Write-Host "  -Restart      Restart the dashboard server"
    Write-Host "  -Status       Check dashboard server status"
    Write-Host "  -Clean        Clean up logs and temporary files"
    Write-Host "  -Environment  Set environment (development|staging|production)"
    Write-Host "  -Port         Set port number (default: 8501)"
    Write-Host "  -Help         Show this help message"
    Write-Host ""
    Write-ColoredOutput "EXAMPLES:" $Yellow
    Write-Host "  .\deploy.ps1 -Install                    # Install dependencies"
    Write-Host "  .\deploy.ps1 -Start                      # Start dashboard"
    Write-Host "  .\deploy.ps1 -Start -Port 8502           # Start on custom port"
    Write-Host "  .\deploy.ps1 -Environment production     # Set production env"
    Write-Host "  .\deploy.ps1 -Restart                    # Restart dashboard"
    Write-Host ""
}

function Test-Prerequisites {
    Write-ColoredOutput "🔍 Checking Prerequisites..." $Blue
    
    # Check Python
    try {
        $PythonVersion = python --version 2>$null
        if ($PythonVersion) {
            Write-ColoredOutput "✅ Python: $PythonVersion" $Green
        } else {
            throw "Python not found"
        }
    } catch {
        Write-ColoredOutput "❌ Python not found. Please install Python 3.8 or higher." $Red
        return $false
    }
    
    # Check pip
    try {
        $PipVersion = pip --version 2>$null
        if ($PipVersion) {
            Write-ColoredOutput "✅ pip: Available" $Green
        } else {
            throw "pip not found"
        }
    } catch {
        Write-ColoredOutput "❌ pip not found. Please install pip." $Red
        return $false
    }
    
    # Check virtual environment
    if (Test-Path "$ProjectRoot\..\.venv\Scripts\activate.ps1") {
        Write-ColoredOutput "✅ Virtual Environment: Found" $Green
        return $true
    } else {
        Write-ColoredOutput "⚠️  Virtual Environment: Not found, will create one" $Yellow
        return $true
    }
}

function Install-Dependencies {
    Write-ColoredOutput "📦 Installing Dependencies..." $Blue
    
    # Create virtual environment if it doesn't exist
    if (!(Test-Path "$ProjectRoot\..\.venv")) {
        Write-ColoredOutput "Creating virtual environment..." $Yellow
        python -m venv "$ProjectRoot\..\.venv"
        if ($LASTEXITCODE -ne 0) {
            Write-ColoredOutput "❌ Failed to create virtual environment" $Red
            return $false
        }
    }
    
    # Activate virtual environment
    & "$ProjectRoot\..\.venv\Scripts\Activate.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-ColoredOutput "❌ Failed to activate virtual environment" $Red
        return $false
    }
    
    Write-ColoredOutput "✅ Virtual environment activated" $Green
    
    # Upgrade pip
    Write-ColoredOutput "Upgrading pip..." $Yellow
    python -m pip install --upgrade pip
    
    # Install core requirements
    if (Test-Path $RequirementsFile) {
        Write-ColoredOutput "Installing core requirements..." $Yellow
        pip install -r $RequirementsFile
        if ($LASTEXITCODE -ne 0) {
            Write-ColoredOutput "❌ Failed to install core requirements" $Red
            return $false
        }
    }
    
    # Install dashboard-specific requirements
    Write-ColoredOutput "Installing dashboard requirements..." $Yellow
    $DashboardReqs = @(
        "streamlit>=1.28.0",
        "plotly>=5.17.0", 
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "redis>=5.0.0",
        "aioredis>=2.0.0",
        "websockets>=11.0.0",
        "PyJWT>=2.8.0",
        "python-dotenv>=1.0.0"
    )
    
    foreach ($req in $DashboardReqs) {
        pip install $req
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColoredOutput "✅ Dependencies installed successfully" $Green
        return $true
    } else {
        Write-ColoredOutput "❌ Failed to install dependencies" $Red
        return $false
    }
}

function Update-Dependencies {
    Write-ColoredOutput "🔄 Updating Dependencies..." $Blue
    
    # Activate virtual environment
    & "$ProjectRoot\..\.venv\Scripts\Activate.ps1"
    
    # Update all packages
    pip install --upgrade streamlit plotly pandas numpy redis aioredis websockets PyJWT python-dotenv
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColoredOutput "✅ Dependencies updated successfully" $Green
        return $true
    } else {
        Write-ColoredOutput "❌ Failed to update dependencies" $Red
        return $false
    }
}

function Start-Dashboard {
    Write-ColoredOutput "🚀 Starting Dashboard Server..." $Blue
    
    # Check if already running
    if (Get-DashboardStatus) {
        Write-ColoredOutput "⚠️  Dashboard is already running" $Yellow
        return $true
    }
    
    # Create logs directory
    $LogDir = Split-Path -Parent $LogFile
    if (!(Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    
    # Set environment variables
    $env:STREAMLIT_SERVER_PORT = $Port
    $env:STREAMLIT_SERVER_ADDRESS = "0.0.0.0"
    $env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
    $env:DASHBOARD_ENVIRONMENT = $Environment
    
    # Activate virtual environment
    & "$ProjectRoot\..\.venv\Scripts\Activate.ps1"
    
    # Start Streamlit dashboard
    Write-ColoredOutput "Starting dashboard on port $Port..." $Yellow
    Write-ColoredOutput "Environment: $Environment" $Yellow
    
    $Process = Start-Process -FilePath "streamlit" -ArgumentList @(
        "run", 
        $MainDashboardFile,
        "--server.port", $Port,
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#2a5298",
        "--theme.backgroundColor", "#ffffff",
        "--theme.secondaryBackgroundColor", "#f0f2f6"
    ) -PassThru -WindowStyle Hidden
    
    # Save process ID
    $Process.Id | Out-File -FilePath $PidFile -Encoding ASCII
    
    # Wait a moment and check if process is still running
    Start-Sleep -Seconds 3
    
    if (Get-Process -Id $Process.Id -ErrorAction SilentlyContinue) {
        Write-ColoredOutput "✅ Dashboard started successfully!" $Green
        Write-ColoredOutput "🌐 Access dashboard at: http://localhost:$Port" $Cyan
        Write-ColoredOutput "📊 Dashboard PID: $($Process.Id)" $Blue
        return $true
    } else {
        Write-ColoredOutput "❌ Failed to start dashboard" $Red
        return $false
    }
}

function Stop-Dashboard {
    Write-ColoredOutput "🛑 Stopping Dashboard Server..." $Blue
    
    if (Test-Path $PidFile) {
        $ProcessId = Get-Content $PidFile -ErrorAction SilentlyContinue
        
        if ($ProcessId) {
            try {
                $Process = Get-Process -Id $ProcessId -ErrorAction Stop
                $Process | Stop-Process -Force
                Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
                Write-ColoredOutput "✅ Dashboard stopped successfully" $Green
                return $true
            } catch {
                Write-ColoredOutput "⚠️  Process not found or already stopped" $Yellow
                Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
                return $true
            }
        }
    }
    
    # Try to find and kill streamlit processes
    $StreamlitProcesses = Get-Process -Name "streamlit" -ErrorAction SilentlyContinue
    if ($StreamlitProcesses) {
        $StreamlitProcesses | Stop-Process -Force
        Write-ColoredOutput "✅ Streamlit processes stopped" $Green
    } else {
        Write-ColoredOutput "ℹ️  No dashboard processes found running" $Blue
    }
    
    return $true
}

function Get-DashboardStatus {
    if (Test-Path $PidFile) {
        $ProcessId = Get-Content $PidFile -ErrorAction SilentlyContinue
        
        if ($ProcessId) {
            try {
                $Process = Get-Process -Id $ProcessId -ErrorAction Stop
                return $true
            } catch {
                Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
                return $false
            }
        }
    }
    
    # Check for any streamlit processes
    $StreamlitProcesses = Get-Process -Name "streamlit" -ErrorAction SilentlyContinue
    return ($StreamlitProcesses -ne $null)
}

function Show-Status {
    Write-ColoredOutput "📊 Dashboard Status" $Cyan
    Write-ColoredOutput "=" * 30 $Blue
    
    if (Get-DashboardStatus) {
        Write-ColoredOutput "Status: 🟢 RUNNING" $Green
        
        if (Test-Path $PidFile) {
            $ProcessId = Get-Content $PidFile -ErrorAction SilentlyContinue
            if ($ProcessId) {
                $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
                if ($Process) {
                    Write-ColoredOutput "PID: $ProcessId" $Blue
                    Write-ColoredOutput "Started: $($Process.StartTime)" $Blue
                    Write-ColoredOutput "CPU Time: $($Process.TotalProcessorTime)" $Blue
                    Write-ColoredOutput "Memory: $([math]::Round($Process.WorkingSet64 / 1MB, 2)) MB" $Blue
                }
            }
        }
        
        Write-ColoredOutput "URL: http://localhost:$Port" $Cyan
    } else {
        Write-ColoredOutput "Status: 🔴 STOPPED" $Red
    }
    
    Write-Host ""
    Write-ColoredOutput "Environment: $Environment" $Blue
    Write-ColoredOutput "Port: $Port" $Blue
    Write-ColoredOutput "Project Root: $ProjectRoot" $Blue
    
    # Check log file
    if (Test-Path $LogFile) {
        $LogInfo = Get-Item $LogFile
        Write-ColoredOutput "Log File: $($LogInfo.FullName)" $Blue
        Write-ColoredOutput "Log Size: $([math]::Round($LogInfo.Length / 1KB, 2)) KB" $Blue
        Write-ColoredOutput "Last Modified: $($LogInfo.LastWriteTime)" $Blue
    }
}

function Clean-Environment {
    Write-ColoredOutput "🧹 Cleaning Environment..." $Blue
    
    # Stop dashboard if running
    Stop-Dashboard | Out-Null
    
    # Clean log files
    if (Test-Path "$ProjectRoot\logs") {
        Remove-Item "$ProjectRoot\logs\*" -Force -Recurse -ErrorAction SilentlyContinue
        Write-ColoredOutput "✅ Cleaned log files" $Green
    }
    
    # Clean temporary files
    $TempFiles = @(
        "$ProjectRoot\*.pyc",
        "$ProjectRoot\__pycache__",
        "$ProjectRoot\*.tmp",
        "$ProjectRoot\.streamlit"
    )
    
    foreach ($Pattern in $TempFiles) {
        Remove-Item $Pattern -Force -Recurse -ErrorAction SilentlyContinue
    }
    
    Write-ColoredOutput "✅ Environment cleaned" $Green
}

function Restart-Dashboard {
    Write-ColoredOutput "🔄 Restarting Dashboard..." $Blue
    
    Stop-Dashboard | Out-Null
    Start-Sleep -Seconds 2
    Start-Dashboard
}

# Main execution logic
Write-ColoredOutput "🚀 $DashboardName Deployment Script" $Magenta
Write-ColoredOutput "=" * 60 $Blue
Write-Host ""

if ($Help) {
    Show-Help
    exit 0
}

if (!(Test-Prerequisites)) {
    Write-ColoredOutput "❌ Prerequisites check failed" $Red
    exit 1
}

# Handle command line arguments
$ActionTaken = $false

if ($Install) {
    Install-Dependencies
    $ActionTaken = $true
}

if ($Update) {
    Update-Dependencies
    $ActionTaken = $true
}

if ($Clean) {
    Clean-Environment
    $ActionTaken = $true
}

if ($Stop) {
    Stop-Dashboard
    $ActionTaken = $true
}

if ($Start) {
    Start-Dashboard
    $ActionTaken = $true
}

if ($Restart) {
    Restart-Dashboard
    $ActionTaken = $true
}

if ($Status) {
    Show-Status
    $ActionTaken = $true
}

# If no action specified, show help
if (!$ActionTaken) {
    Write-ColoredOutput "⚠️  No action specified" $Yellow
    Write-Host ""
    Show-Help
}

Write-Host ""
Write-ColoredOutput "✨ Script completed" $Green