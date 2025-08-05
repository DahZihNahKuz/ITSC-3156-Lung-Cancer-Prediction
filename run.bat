@echo off
rem A basic script to set up a virtual environment, install dependencies,
rem and run the Flask server on Windows.

echo Starting the application setup...

rem Step 1: Create a virtual environment named 'venv'
echo Creating virtual environment...
python -m venv venv

rem Step 2: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

rem Step 3: Install all packages from requirements.txt
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

rem Step 4: Run the Flask application
echo Starting Flask server.
echo This window must remain open to control the server.
echo Press Ctrl+C to stop the server when you are finished.

rem Step 5: Open the HTML file in the default web browser
echo Opening the web form in your browser...
start "" "Lung Cancer Prediction Web Form.html"

python app.py