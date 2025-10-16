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
EYE_PARTIAL_THRESH = 0.40   # threshold for partial eye closure detection
MICROSLEEP_SEC    = 1.5     # eyes closed for this long -> microsleep
PERCLOS_WIN_SEC   = 60.0    # rolling window for PERCLOS
PERCLOS_DROWSY    = 0.70    # >=70% of last minute eyes-closed -> drowsy
YAWN_THRESH       = 0.80    # mouthLowerDown above this = yawning
YAWN_MIN_SEC      = 1.0     # sustained open for this long -> yawn flag
EYE_CLOSURE_FREQ_WIN = 15.0  # 15 seconds window for eye closure frequency
EYE_CLOSURE_FREQ_THRESH = 4  # more than 4 closures in 10 seconds = drowsy
MIN_CLOSURE_DURATION = 0.4   # minimum duration (seconds) to count as drowsy closure, not blink

# ---- State (globals) ----
EYE_CLOSED_START = None
YAWN_START = None
PERCLOS_WIN = deque()   # holds (timestamp, is_closed_int)
BLINK_TIMES = deque()   # timestamps of blinks in the last 60s
EYE_CLOSURE_EVENTS = deque()  # timestamps of eye closures > 0.4 in last 10s
EYE_PARTIAL_CLOSURE_START = None  # track when partial closure (>0.4) starts

# Event counters
YAWN_COUNT = 0
DROWSY_COUNT = 0
MICROSLEEP_COUNT = 0
YAWN_COUNTED = False  # Flag to prevent counting same yawn multiple times
MICROSLEEP_COUNTED = False  # Flag to prevent counting same microsleep multiple times
DROWSY_COUNTED = False  # Flag to prevent counting same drowsy event multiple times

# Add these global variables at the top with other globals
SCROLL_OFFSET = 0
MAX_SCROLL = 0

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

        # Use mouthLowerDownRight and mouthLowerDownLeft for yawn detection
        mouth_lower_down_r = _bs_score(bs, "mouthLowerDownRight")
        mouth_lower_down_l = _bs_score(bs, "mouthLowerDownLeft")
        mouth_lower_down = 0.5 * (mouth_lower_down_r + mouth_lower_down_l)

        # --- PERCLOS window maintenance (1 = closed, 0 = open) ---
        is_closed = 1 if eye_closed_score > EYE_CLOSED_THRESH else 0
        PERCLOS_WIN.append((now, is_closed))
        while PERCLOS_WIN and (now - PERCLOS_WIN[0][0]) > PERCLOS_WIN_SEC:
            PERCLOS_WIN.popleft()
        perclos = (sum(v for _, v in PERCLOS_WIN) / len(PERCLOS_WIN)) if PERCLOS_WIN else 0.0

        # --- Eye closure frequency tracking (>0.4 threshold, duration >= 0.4s) ---
        global EYE_CLOSURE_EVENTS, EYE_PARTIAL_CLOSURE_START
        
        # Track when partial closure starts
        if eye_closed_score > EYE_PARTIAL_THRESH:
            if EYE_PARTIAL_CLOSURE_START is None:
                EYE_PARTIAL_CLOSURE_START = now
        else:
            # Eyes reopened - check if this was a drowsy closure (not a quick blink)
            if EYE_PARTIAL_CLOSURE_START is not None:
                closure_duration = now - EYE_PARTIAL_CLOSURE_START
                
                # Only count if closure lasted at least MIN_CLOSURE_DURATION (0.4 seconds)
                # This filters out quick blinks
                if closure_duration >= MIN_CLOSURE_DURATION:
                    # Avoid counting the same closure multiple times
                    if not EYE_CLOSURE_EVENTS or (EYE_PARTIAL_CLOSURE_START - EYE_CLOSURE_EVENTS[-1]) > 0.5:
                        EYE_CLOSURE_EVENTS.append(now)
                
                EYE_PARTIAL_CLOSURE_START = None
        
        # Keep only last 10 seconds of closure events
        while EYE_CLOSURE_EVENTS and (now - EYE_CLOSURE_EVENTS[0]) > EYE_CLOSURE_FREQ_WIN:
            EYE_CLOSURE_EVENTS.popleft()
        
        # Check if frequency exceeds threshold
        frequent_closures = len(EYE_CLOSURE_EVENTS) > EYE_CLOSURE_FREQ_THRESH

        # --- Microsleep & blink counting ---
        global EYE_CLOSED_START, MICROSLEEP_COUNT, MICROSLEEP_COUNTED
        if is_closed:
            if EYE_CLOSED_START is None:
                EYE_CLOSED_START = now
        else:
            # eye reopened -> count a blink if it was brief
            if EYE_CLOSED_START is not None:
                duration = now - EYE_CLOSED_START
                if duration < 0.4:   # Quick blink (less than 0.4 seconds)
                    BLINK_TIMES.append(now)
                EYE_CLOSED_START = None
                MICROSLEEP_COUNTED = False  # Reset flag when eyes reopen

        # keep only last 60s blinks
        while BLINK_TIMES and (now - BLINK_TIMES[0]) > 60.0:
            BLINK_TIMES.popleft()
        blinks_per_min = len(BLINK_TIMES)

        microsleep = (EYE_CLOSED_START is not None) and ((now - EYE_CLOSED_START) >= MICROSLEEP_SEC)
        
        # Count microsleep event only once
        if microsleep and not MICROSLEEP_COUNTED:
            MICROSLEEP_COUNT += 1
            MICROSLEEP_COUNTED = True

        # --- Yawn detection (sustained high mouthLowerDown) ---
        global YAWN_START, YAWN_COUNT, YAWN_COUNTED
        yawning = False
        if mouth_lower_down > YAWN_THRESH:
            if YAWN_START is None:
                YAWN_START = now
            elif (now - YAWN_START) >= YAWN_MIN_SEC:
                yawning = True
                # Count yawn only once per event
                if not YAWN_COUNTED:
                    YAWN_COUNT += 1
                    YAWN_COUNTED = True
        else:
            YAWN_START = None
            YAWN_COUNTED = False  # Reset flag when yawn ends

        # --- Drowsiness decision ---
        global DROWSY_COUNT, DROWSY_COUNTED
        drowsy = microsleep or (perclos >= PERCLOS_DROWSY) or yawning or frequent_closures
        
        # Count drowsiness event only once per continuous drowsy period
        if drowsy and not DROWSY_COUNTED:
            DROWSY_COUNT += 1
            DROWSY_COUNTED = True
        elif not drowsy:
            DROWSY_COUNTED = False  # Reset flag when alert

        # Return the metrics for display
        return {
            'drowsy': drowsy,
            'yawning': yawning,
            'microsleep': microsleep,
            'perclos': perclos,
            'blinks_per_min': blinks_per_min,
            'frequent_closures': frequent_closures,
            'closure_count': len(EYE_CLOSURE_EVENTS),
            'yawn_count': YAWN_COUNT,
            'drowsy_count': DROWSY_COUNT,
            'microsleep_count': MICROSLEEP_COUNT
        }
    return None

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

    global SCROLL_OFFSET, MAX_SCROLL

    # Mouse callback for scrolling
    def mouse_callback(event, x, y, flags, param):
        global SCROLL_OFFSET, MAX_SCROLL
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # Scroll up
                SCROLL_OFFSET = max(0, SCROLL_OFFSET - 20)
            else:  # Scroll down
                SCROLL_OFFSET = min(MAX_SCROLL, SCROLL_OFFSET + 20)

    cv2.namedWindow('SafeDriver Monitoring System')
    cv2.setMouseCallback('SafeDriver Monitoring System', mouse_callback)

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

        # Show the FPS with smaller font
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size - 20)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location,
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5, text_color, 1, cv2.LINE_AA)  # Reduced size and thickness

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
            legend_x = current_frame.shape[1] - label_padding_width + 20
            legend_y = 30 - SCROLL_OFFSET
            bar_max_width = label_padding_width - 40
            bar_height = 8
            gap_between_bars = 5
            text_gap = 5

            face_blendshapes = DETECTION_RESULT.face_blendshapes

            if face_blendshapes:

                # Detect driver behavior
                behavior_data = detect_driver_behavior(face_blendshapes, current_frame.shape[0], current_frame)
                
                if behavior_data:
                    # Calculate the background rectangle dimensions
                    metrics_padding = 10  # pixels padding around text
                    metrics_x = left_margin - metrics_padding
                    metrics_y = row_size # Moved up by 10px (was row_size + 20)
                    metrics_width = 175  # Width to accommodate all text
                    metrics_height = 140  # Height for 6 metrics
                    
                    # Create a semi-transparent white overlay for the metrics
                    overlay = current_frame.copy()
                    
                    # Draw rounded rectangle (using multiple rectangles and circles for corners)
                    corner_radius = 10
                    
                    # Main rectangles - adjusted to align with corner circles
                    cv2.rectangle(overlay,
                                (metrics_x + corner_radius, metrics_y),
                                (metrics_x + metrics_width - corner_radius, metrics_y + metrics_height),
                                (255, 255, 255), -1)
                    cv2.rectangle(overlay,
                                (metrics_x, metrics_y + corner_radius),
                                (metrics_x + metrics_width, metrics_y + metrics_height - corner_radius),
                                (255, 255, 255), -1)
                    
                    # Corner circles - keep same positions
                    cv2.circle(overlay, (metrics_x + corner_radius, metrics_y + corner_radius),
                             corner_radius, (255, 255, 255), -1)
                    cv2.circle(overlay, (metrics_x + metrics_width - corner_radius, metrics_y + corner_radius),
                             corner_radius, (255, 255, 255), -1)
                    cv2.circle(overlay, (metrics_x + corner_radius, metrics_y + metrics_height - corner_radius),
                             corner_radius, (255, 255, 255), -1)
                    cv2.circle(overlay, (metrics_x + metrics_width - corner_radius, metrics_y + metrics_height - corner_radius),
                             corner_radius, (255, 255, 255), -1)
                    
                    # Apply the overlay with 0.5 opacity
                    cv2.addWeighted(overlay, 0.5, current_frame, 0.5, 0, current_frame)
                    
                    # Display PERCLOS and Blink rate under FPS (top left)
                    cv2.putText(current_frame, f"PERCLOS: {behavior_data['perclos']:.2f}", 
                               (left_margin, row_size + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(current_frame, f"Blinks/min: {behavior_data['blinks_per_min']:02d}", 
                               (left_margin, row_size + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(current_frame, f"Closures(15s): {behavior_data['closure_count']}", 
                               (left_margin, row_size + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    # Display event counts (top left, below other metrics)
                    cv2.putText(current_frame, f"Yawns: {behavior_data['yawn_count']}", 
                               (left_margin, row_size + 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(current_frame, f"Microsleeps: {behavior_data['microsleep_count']}", 
                               (left_margin, row_size + 105),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(current_frame, f"Drowsy Events: {behavior_data['drowsy_count']}", 
                               (left_margin, row_size + 125),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    # Display warnings in top right (only show if detected)
                    frame_width = current_frame.shape[1] - label_padding_width
                    warning_color = (0, 0, 255)  # Red color
                    
                    # Priority: Microsleep > Yawning > Frequent Closures > General Drowsiness
                    if behavior_data['microsleep']:
                        warning_text = "Microsleep Detected!"
                        (text_width, text_height), _ = cv2.getTextSize(warning_text, 
                                                                        cv2.FONT_HERSHEY_DUPLEX, 
                                                                        1.0, 2)
                        right_x = frame_width - text_width - 20
                        cv2.putText(current_frame, warning_text, 
                                   (right_x, 50),
                                   cv2.FONT_HERSHEY_DUPLEX, 1.0, warning_color, 2, cv2.LINE_AA)
                        print(f"Microsleep detected (Total: {behavior_data['microsleep_count']})")
                    
                    elif behavior_data['yawning']:
                        warning_text = "Yawning Detected!"
                        (text_width, text_height), _ = cv2.getTextSize(warning_text, 
                                                                        cv2.FONT_HERSHEY_DUPLEX, 
                                                                        1.0, 2)
                        right_x = frame_width - text_width - 20
                        cv2.putText(current_frame, warning_text, 
                                   (right_x, 50),
                                   cv2.FONT_HERSHEY_DUPLEX, 1.0, warning_color, 2, cv2.LINE_AA)
                        print(f"Yawn detected (Total: {behavior_data['yawn_count']})")
                    
                    elif behavior_data['frequent_closures']:
                        warning_text = "Frequent Eye Closures!"
                        (text_width, text_height), _ = cv2.getTextSize(warning_text, 
                                                                        cv2.FONT_HERSHEY_DUPLEX, 
                                                                        1.0, 2)
                        right_x = frame_width - text_width - 20
                        cv2.putText(current_frame, warning_text, 
                                   (right_x, 50),
                                   cv2.FONT_HERSHEY_DUPLEX, 1.0, warning_color, 2, cv2.LINE_AA)
                        print("Frequent eye closures detected")
                    
                    elif behavior_data['drowsy']:
                        warning_text = "Drowsiness Detected!"
                        (text_width, text_height), _ = cv2.getTextSize(warning_text, 
                                                                        cv2.FONT_HERSHEY_DUPLEX, 
                                                                        1.0, 2)
                        right_x = frame_width - text_width - 20
                        cv2.putText(current_frame, warning_text, 
                                   (right_x, 50),
                                   cv2.FONT_HERSHEY_DUPLEX, 1.0, warning_color, 2, cv2.LINE_AA)
                        print(f"Drowsiness detected (Total: {behavior_data['drowsy_count']})")
                
                num_blendshapes = len(face_blendshapes[0])
                total_height = num_blendshapes * (bar_height + gap_between_bars)
                MAX_SCROLL = max(0, total_height - current_frame.shape[0] + 60)
                
                for idx, category in enumerate(face_blendshapes[0]):
                    # Only draw if within visible area
                    if legend_y + bar_height > 0 and legend_y < current_frame.shape[0]:
                        category_name = category.category_name
                        score = round(category.score, 2)

                        # Prepare text and get its width
                        text = "{} ({:.2f})".format(category_name, score)
                        (text_width, _), _ = cv2.getTextSize(text,
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            0.4, 1)

                        # Display the blendshape name and score
                        cv2.putText(current_frame, text,
                                    (legend_x, legend_y + (bar_height // 2) + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (0, 0, 0), 1, cv2.LINE_AA)

                        # Calculate bar width based on score
                        bar_width = int(bar_max_width * score)

                        # Draw the bar to the right of the text
                        cv2.rectangle(current_frame,
                                        (legend_x + text_width + text_gap, legend_y),
                                        (legend_x + text_width + text_gap + bar_width,
                                        legend_y + bar_height),
                                        (0, 255, 0), -1)

                    legend_y += (bar_height + gap_between_bars)

        cv2.imshow('SafeDriver Monitoring System', current_frame)

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