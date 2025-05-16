@echo off
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
echo Environment setup complete!