import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import face_recognition
from scipy.spatial import distance

import warnings
warnings.filterwarnings('ignore')

image_path = 'test1.png'
image = Image.open(image_path)
# plt.axis('off')
# plt.imshow(image)
# plt.show()

def highlight_facial_points(image_path):
    # load the image
    image_bgr = cv2.imread(image_path)
    # convert from bgr to rgb
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # detect faces in the image
    face_locations=face_recognition.face_locations(image_rgb, model='hog')

    for face_location in face_locations:
        # get facial landmarks
        landmarks = face_recognition.face_landmarks(image_rgb, [face_location])[0]

        # Iterate over the facial landmarks and draw them on the image
        for landmark_type, landmark_points in landmarks.items():
            for (x, y) in landmark_points:
                cv2.circle(image_rgb, (x, y), 3, (0, 255, 0), -1)

    # plot the image
    # plt.figure(figsize=(6, 6))
    # plt.imshow(image_rgb)
    # plt.axis('off')
    # plt.show()

highlight_facial_points(image_path)

# calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A+B) / (2.0 * C)
    return ear

# calculate mount aspect ratio
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A+B) / (2.0 * C)
    return mar

# Resize the image before processing to improve performance
def process_image(frame):
    # Resize frame to smaller dimensions for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert BGR to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # find face locations on smaller frame
    face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
    
    # Scale back up face locations
    face_locations_full_size = [(top*4, right*4, bottom*4, left*4) 
                              for top, right, bottom, left in face_locations]
    
    # define thresholds
    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.6

    if frame is None:
        raise ValueError('Image is not found or unable to open')

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # initiate flags
    eye_flag = mouth_flag = False

    # Return early if no faces found
    if not face_locations_full_size:
        return eye_flag, mouth_flag

    for face_location in face_locations_full_size:
        try:
            # extract facial landmarks
            landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])
            if not landmarks:
                continue
                
            landmarks = landmarks[0]
            
            # Check if all required facial features are detected
            if not all(k in landmarks for k in ['left_eye', 'right_eye', 'bottom_lip']):
                continue
                
            # extract eye and mouth coordinates
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            mouth = np.array(landmarks['bottom_lip'])

            # calculate ear and mar
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear+right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # check if eyes are closed
            if ear < EYE_AR_THRESH:
                eye_flag = True

            # check if yawning
            if mar > MOUTH_AR_THRESH:
                mouth_flag = True

        except Exception as e:
            print(f"Error processing landmarks: {e}")
            continue

    return eye_flag, mouth_flag

img = cv2.imread(image_path)
process_image(img)

# Try webcam first, fallback to video file if not available
try:
    video_cap = cv2.VideoCapture(0)
    if not video_cap.isOpened():
        print("Could not open webcam, trying video file...")
        video_path = "test.mp4"  # Change to your video file
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            raise Exception("Could not open video source")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

count = score = 0

while True:
    success, image = video_cap.read()
    if not success:
        break

    image = cv2.resize(image, (800, 500))

    count += 1
    # process every nth frame
    n = 5
    if count % n == 0:
        eye_flag, mouth_flag = process_image(image)
        # if any flag is true, increment the score
        if eye_flag or mouth_flag:
            score += 1
        else:
            score -= 1
            if score < 0:
                score = 0

    # write the score values at bottom left of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_x = 10
    text_y = image.shape[0] - 10
    text = f"Score: {score}"
    cv2.putText(image, text, (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if score >= 5:
        text_x = image.shape[1] - 130
        text_y = 40
        text = "Drowsy"
        cv2.putText(image, text, (text_x, text_y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('drowsiness detection', image)

    # exit if any key is pressed
    if cv2.waitKey(1) & 0xFF != 255:
        break

video_cap.release()
cv2.destroyAllWindows()