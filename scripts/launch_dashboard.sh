#!/bin/bash
# Launch rfx Streamlit dashboard on VESSL interactive pod
# Usage: bash scripts/launch_dashboard.sh
#
# Access: After launching, the dashboard will be available at
# the VESSL pod's exposed port (default 8501).
# In VESSL web terminal, the URL is shown after startup.

set -e

echo "=== rfx Dashboard Launcher ==="

# Install dependencies if needed
pip install streamlit matplotlib pillow -q 2>/dev/null

# Install rfx in development mode
pip install -e ".[all]" -q 2>/dev/null || pip install -e . -q 2>/dev/null

echo ""
echo "Starting rfx dashboard on port 8501..."
echo "Access URL: http://localhost:8501"
echo ""

# Run streamlit with VESSL-friendly settings
streamlit run rfx/dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#1f77b4" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f0f2f6"
