# safe-driver

### install python version 3.10.9
        https://www.python.org/downloads/release/python-3109/

### install python extension to vs-code
        Python by Microsoft microsoft.com

### check python version [3.10.9]
        python --version

## check the environment variables to ensure the path is correct

### create python environment
        python -m venv venv
        .\venv\Scripts\Activate

## - Temporarily Change Execution Policy
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

## - Permenantly Change Execution Policy
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

## 

### Upgrade pip (Important)
        pip install --upgrade pip

### install (or update) the project libraries
        pip install -r requirements.txt

## DEVELOPER MODE INSTRUCTIONS [DO NOT RUN BELLOW]

### download dlib library
        https://github.com/z-mahmud22/Dlib_Windows_Python3.x/blob/main/dlib-19.22.99-cp310-cp310-win_amd64.whl

### copy the dlib library to 
        path --> safe-driver-model/

## install libraries
        pip install mediapipe opencv-python numpy gps3
        pip install cmake
        pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
        pip install face_recognition

### save the working environment (run only when new libraries installed)
        pip freeze > requirements.txt

