#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Start the separate service in the background
python src/run_service.py &

# Start the Streamlit app
streamlit run src/streamlit_app.py --server.port $PORT --server.enableCORS false
