#!/bin/bash
# Simple script to run the Redubber Streamlit application

echo "Starting Redubber - Audio Redub Project Manager..."
echo "The application will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

cd "$(dirname "$0")"
streamlit run app.py