
# =================================================================================
                                    safe-driver
# =================================================================================

# install python version 3.10.9
https://www.python.org/downloads/release/python-3109/

# install python extension to vs-code
Python by Microsoft microsoft.com

# check python version [3.10.9]
python --version

# create python environment
python -m venv venv
.\venv\Scripts\Activate

#   - Temporarily Change Execution Policy
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

#   - Permenantly Change Execution Policy
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# =================================================================================

# Upgrade pip (Important)
pip install --upgrade pip

# install the project libraries
pip install -r requirements.txt

# ============================[DO_NOT_RUN_BELLOW]==================================

# libraries
pip install mediapipe opencv-python numpy gps3

pip install dlib-19.22.99-cp310-cp310-win_amd64.whl

# save the working environment
pip freeze > requirements.txt

