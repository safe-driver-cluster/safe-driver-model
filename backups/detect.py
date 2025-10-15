# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main scripts to run face landmarker."""

import argparse
import sys
import time

import numpy as np
from collections import deque

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

# ---- Drowsiness thresholds (tune for your camera/person/environment) ----
EYE_CLOSED_THRESH = 0.60    # avg(eyeBlinkLeft, eyeBlinkRight) above this = closed
MICROSLEEP_SEC    = 1.5     # eyes closed for this long -> microsleep
PERCLOS_WIN_SEC   = 60.0    # rolling window for PERCLOS
PERCLOS_DROWSY    = 0.70    # >=70% of last minute eyes-closed -> drowsy
YAWN_THRESH       = 0.40    # jawOpen above this = mouth open
YAWN_MIN_SEC      = 1.0     # sustained open for this long -> yawn flag

# ---- State (globals) ----
EYE_CLOSED_START = None
YAWN_START = None
PERCLOS_WIN = deque()   # holds (timestamp, is_closed_int)
BLINK_TIMES = deque()   # timestamps of blinks in the last 60s

def _bs_score(blendshapes, name: str) -> float:
    """Return blendshape score by name or 0.0 if missing."""
    for c in blendshapes:
        if c.category_name == name:
            return float(c.score)
    return 0.0

def detect_driver_behavior(face_blendshapes: np.ndarray, height, current_frame) -> str:
    if face_blendshapes:
        now = time.time()
        bs = face_blendshapes[0]

        # --- Eye & mouth signals from blendshapes ---
        blink_l = _bs_score(bs, "eyeBlinkLeft")
        blink_r = _bs_score(bs, "eyeBlinkRight")
        eye_closed_score = 0.5 * (blink_l + blink_r)

        jaw_open = _bs_score(bs, "jawOpen")

        # --- PERCLOS window maintenance (1 = closed, 0 = open) ---
        is_closed = 1 if eye_closed_score > EYE_CLOSED_THRESH else 0
        PERCLOS_WIN.append((now, is_closed))
        while PERCLOS_WIN and (now - PERCLOS_WIN[0][0]) > PERCLOS_WIN_SEC:
            PERCLOS_WIN.popleft()
        perclos = (sum(v for _, v in PERCLOS_WIN) / len(PERCLOS_WIN)) if PERCLOS_WIN else 0.0

        # --- Microsleep & blink counting ---
        global EYE_CLOSED_START
        if is_closed:
            if EYE_CLOSED_START is None:
                EYE_CLOSED_START = now
        else:
            # eye reopened -> count a blink if it was brief
            if EYE_CLOSED_START is not None:
                duration = now - EYE_CLOSED_START
                if duration < 0.8:   # blink (tune)
                    BLINK_TIMES.append(now)
                EYE_CLOSED_START = None

        # keep only last 60s blinks
        while BLINK_TIMES and (now - BLINK_TIMES[0]) > 60.0:
            BLINK_TIMES.popleft()
        blinks_per_min = len(BLINK_TIMES)

        microsleep = (EYE_CLOSED_START is not None) and ((now - EYE_CLOSED_START) >= MICROSLEEP_SEC)

        # --- Yawn detection (sustained high jawOpen) ---
        global YAWN_START
        yawning = False
        if jaw_open > YAWN_THRESH:
            if YAWN_START is None:
                YAWN_START = now
            elif (now - YAWN_START) >= YAWN_MIN_SEC:
                yawning = True
        else:
            YAWN_START = None

        # --- Drowsiness decision ---
        drowsy = microsleep or (perclos >= PERCLOS_DROWSY) or yawning

        # 2. Second change - in detect_driver_behavior() function, modify the panel positioning:
        # --- On-screen status panel ---
        panel_w = 460
        panel_h = 90
        # Ensure panel stays within the frame bounds
        y0 = min(height - panel_h - 10, height - 1 - panel_h)
        x0 = 10
        cv2.rectangle(current_frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), -1)
        
        status_text = "DROWSY" if drowsy else "OK"
        status_color = (0, 0, 255) if drowsy else (0, 150, 0)

        cv2.putText(current_frame, f"Status: {status_text}", (x0 + 12, y0 + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, status_color, 2, cv2.LINE_AA)
        cv2.putText(current_frame, f"PERCLOS: {perclos:.2f}   Blinks/min: {blinks_per_min:02d}",
                    (x0 + 12, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        if yawning:
            cv2.putText(current_frame, "Yawn", (x0 + 280, y0 + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA)
            print("Yawn detected")
        if microsleep:
            cv2.putText(current_frame, "Microsleep!", (x0 + 340, y0 + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2, cv2.LINE_AA)
            print("Microsleep detected")

def detect_blenshapes(category_name, score, current_frame):
    # Example: Print or log specific blendshape detections
    if category_name == "mouthOpen" and score > 0.5:
        print(f"Mouth open detected with score: {score:.2f}")
    elif category_name == "eyeBlinkLeft" and score > 0.5:
        print(f"Left eye blink detected with score: {score:.2f}")
    elif category_name == "eyeBlinkRight" and score > 0.5:
        print(f"Right eye blink detected with score: {score:.2f}")
    # Add more conditions as needed for other blendshapes

def run(model: str, num_faces: int,
        min_face_detection_confidence: float,
        min_face_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the face landmarker model bundle.
      num_faces: Max number of faces that can be detected by the landmarker.
      min_face_detection_confidence: The minimum confidence score for face
        detection to be considered successful.
      min_face_presence_confidence: The minimum confidence score of face
        presence score in the face landmark detection.
      min_tracking_confidence: The minimum confidence score for the face
        tracking to be considered successful.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Label box parameters
    label_background_color = (255, 255, 255)  # White
    label_padding_width = 1500  # pixels

    def save_result(result: vision.FaceLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate the FPS (frame per second)
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

    # Initialize the face landmarker model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=num_faces,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=True,
        result_callback=save_result)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run face landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location,
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if DETECTION_RESULT:
            # Draw landmarks.
            for face_landmarks in DETECTION_RESULT.face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x,
                                                    y=landmark.y,
                                                    z=landmark.z) for
                    landmark in
                    face_landmarks
                ])
                mp_drawing.draw_landmarks(
                    image=current_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=current_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=current_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_iris_connections_style())

        # Expand the right side frame to show the blendshapes.
        current_frame = cv2.copyMakeBorder(current_frame, 0, 0, 0,
                                           label_padding_width,
                                           cv2.BORDER_CONSTANT, None,
                                           label_background_color)

        if DETECTION_RESULT:
            # Define parameters for the bars and text
            legend_x = current_frame.shape[
                            1] - label_padding_width + 20  # Starting X-coordinate (20 as a margin)
            legend_y = 30  # Starting Y-coordinate
            bar_max_width = label_padding_width - 40  # Max width of the bar with some margin
            bar_height = 8  # Height of the bar
            gap_between_bars = 5  # Gap between two bars
            text_gap = 5  # Gap between the end of the text and the start of the bar

            face_blendshapes = DETECTION_RESULT.face_blendshapes

            if face_blendshapes:

                # Detect driver behavior
                detect_driver_behavior(face_blendshapes, current_frame.shape[0], current_frame)
                
                for idx, category in enumerate(face_blendshapes[0]):
                    category_name = category.category_name
                    score = round(category.score, 2)

                    detect_blenshapes(category_name, score, current_frame)

                    # Prepare text and get its width
                    text = "{} ({:.2f})".format(category_name, score)
                    (text_width, _), _ = cv2.getTextSize(text,
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.4, 1)

                    # Display the blendshape name and score
                    cv2.putText(current_frame, text,
                                (legend_x, legend_y + (bar_height // 2) + 5),
                                # Position adjusted for vertical centering
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,  # Font size
                                (0, 0, 0),  # Black color
                                1,
                                cv2.LINE_AA)  # Thickness

                    # Calculate bar width based on score
                    bar_width = int(bar_max_width * score)

                    # Draw the bar to the right of the text
                    cv2.rectangle(current_frame,
                                    (legend_x + text_width + text_gap, legend_y),
                                    (legend_x + text_width + text_gap + bar_width,
                                    legend_y + bar_height),
                                    (0, 255, 0),  # Green color
                                    -1)  # Filled bar

                    # Update the Y-coordinate for the next bar
                    legend_y += (bar_height + gap_between_bars)

        cv2.imshow('face_landmarker', current_frame)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of face landmarker model.',
        required=False,
        default='face_landmarker.task')
    parser.add_argument(
        '--numFaces',
        help='Max number of faces that can be detected by the landmarker.',
        required=False,
        default=1)
    parser.add_argument(
        '--minFaceDetectionConfidence',
        help='The minimum confidence score for face detection to be considered '
             'successful.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minFacePresenceConfidence',
        help='The minimum confidence score of face presence score in the face '
             'landmark detection.',
        required=False,
        default=0.5)
    parser.add_argument(
        '--minTrackingConfidence',
        help='The minimum confidence score for the face tracking to be '
             'considered successful.',
        required=False,
        default=0.5)
    # Finding the camera ID can be very reliant on platform-dependent methods.
    # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
    # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
    # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=720)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=960)
    args = parser.parse_args()

    run(args.model, int(args.numFaces), args.minFaceDetectionConfidence,
        args.minFacePresenceConfidence, args.minTrackingConfidence,
        int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()
