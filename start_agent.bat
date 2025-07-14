@echo off
echo Starting Portfolio Copilot System...

REM Start all agents in separate windows
start "Customer Portfolio Agent" cmd /c "cd /d "%~dp0customer-portfolio-agent" && uv sync --frozen && uv run a2a_server.py"
start "Portfolio Analytics Agent" cmd /c "cd /d "%~dp0portfolio-analytics-agent" && uv sync --frozen && uv run a2a_server.py"
start "Risk Calculation Agent" cmd /c "cd /d "%~dp0risk-calculation-agent" && uv sync --frozen && uv run a2a_server.py"

REM Wait for agents to start
echo Waiting for agents to initialize...
timeout /t 15 >nul

REM Start Streamlit
echo Starting Streamlit app...
cd /d "%~dp0copilot-agent"
streamlit run streamlit.py

REM Clean up when done
taskkill /F /FI "WindowTitle eq Customer Portfolio Agent*" >nul 2>&1
taskkill /F /FI "WindowTitle eq Portfolio Analytics Agent*" >nul 2>&1
taskkill /F /FI "WindowTitle eq Risk Calculation Agent*" >nul 2>&1