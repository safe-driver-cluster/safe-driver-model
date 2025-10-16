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

# ============================================================================
# CONFIGURATION SECTION - All configurable parameters
# ============================================================================

CONFIG = {
    # Drowsiness Detection Thresholds
    'EYE_CLOSED_THRESH': 0.60,
    'EYE_PARTIAL_THRESH': 0.40,
    'MICROSLEEP_SEC': 1.5,
    'PERCLOS_WIN_SEC': 60.0,
    'PERCLOS_DROWSY': 0.70,
    'YAWN_THRESH': 0.80,
    'YAWN_MIN_SEC': 1.0,
    'EYE_CLOSURE_FREQ_WIN': 15.0,
    'EYE_CLOSURE_FREQ_THRESH': 4,
    'MIN_CLOSURE_DURATION': 0.4,
    'BLINK_MAX_DURATION': 0.4,
    'CLOSURE_DEBOUNCE_TIME': 0.5,
    
    # UI Layout Parameters
    'WINDOW_NAME': 'SafeDriver Monitoring System',
    'ROW_SIZE': 50,
    'LEFT_MARGIN': 24,
    'LABEL_PADDING_WIDTH': 1500,
    'FPS_AVG_FRAME_COUNT': 10,
    'SCROLL_STEP': 20,
    
    # Display Control Flags
    'SHOW_BLENDSHAPES': False,  # Set to False to hide blendshapes panel
    'SHOW_FACE_MESH': True,    # Set to False to hide face mesh overlay
    'SHOW_FPS': True,          # Set to False to hide FPS counter
    'SHOW_METRICS': True,      # Set to False to hide metrics box
    'SHOW_WARNINGS': True,     # Set to False to hide warning messages
    
    # FPS Display
    'FPS_FONT': cv2.FONT_HERSHEY_DUPLEX,
    'FPS_FONT_SIZE': 0.5,
    'FPS_FONT_THICKNESS': 1,
    'FPS_COLOR': (0, 0, 0),
    'FPS_TEXT_FORMAT': 'FPS = {:.1f}',
    'FPS_Y_OFFSET': -20,
    
    # Metrics Box (Top Left)
    'METRICS_PADDING': 10,
    'METRICS_WIDTH': 175,
    'METRICS_HEIGHT': 140,
    'METRICS_Y_OFFSET': 0,
    'METRICS_CORNER_RADIUS': 10,
    'METRICS_BG_COLOR': (255, 255, 255),
    'METRICS_BG_OPACITY': 0.5,
    'METRICS_FONT': cv2.FONT_HERSHEY_SIMPLEX,
    'METRICS_FONT_SIZE': 0.5,
    'METRICS_FONT_THICKNESS': 1,
    'METRICS_TEXT_COLOR': (0, 0, 0),
    
    # Metrics Text Labels
    'LABEL_PERCLOS': 'PERCLOS: {:.2f}',
    'LABEL_BLINKS': 'Blinks/min: {:02d}',
    'LABEL_CLOSURES': 'Closures(15s): {}',
    'LABEL_YAWNS': 'Yawns: {}',
    'LABEL_MICROSLEEPS': 'Microsleeps: {}',
    'LABEL_DROWSY_EVENTS': 'Drowsy Events: {}',
    
    # Metrics Text Positions (relative to row_size)
    'PERCLOS_Y_OFFSET': 20,
    'BLINKS_Y_OFFSET': 40,
    'CLOSURES_Y_OFFSET': 60,
    'YAWNS_Y_OFFSET': 85,
    'MICROSLEEPS_Y_OFFSET': 105,
    'DROWSY_EVENTS_Y_OFFSET': 125,
    
    # Warning Display (Top Right)
    'WARNING_FONT': cv2.FONT_HERSHEY_DUPLEX,
    'WARNING_FONT_SIZE': 1.0,
    'WARNING_FONT_THICKNESS': 2,
    'WARNING_COLOR': (0, 0, 255),
    'WARNING_Y_POSITION': 50,
    'WARNING_RIGHT_MARGIN': 20,
    
    # Warning Text Messages
    'WARNING_MICROSLEEP': 'Microsleep Detected!',
    'WARNING_YAWNING': 'Yawning Detected!',
    'WARNING_FREQUENT_CLOSURES': 'Frequent Eye Closures!',
    'WARNING_DROWSY': 'Drowsiness Detected!',
    
    # Console Messages
    'CONSOLE_MICROSLEEP': 'Microsleep detected (Total: {})',
    'CONSOLE_YAWN': 'Yawn detected (Total: {})',
    'CONSOLE_FREQUENT_CLOSURES': 'Frequent eye closures detected',
    'CONSOLE_DROWSY': 'Drowsiness detected (Total: {})',
    
    # Blendshapes Display (Right Panel)
    'BLENDSHAPE_FONT': cv2.FONT_HERSHEY_SIMPLEX,
    'BLENDSHAPE_FONT_SIZE': 0.4,
    'BLENDSHAPE_FONT_THICKNESS': 1,
    'BLENDSHAPE_TEXT_COLOR': (0, 0, 0),
    'BLENDSHAPE_BAR_COLOR': (0, 255, 0),
    'BLENDSHAPE_BAR_HEIGHT': 8,
    'BLENDSHAPE_GAP_BETWEEN_BARS': 5,
    'BLENDSHAPE_TEXT_GAP': 5,
    'BLENDSHAPE_X_OFFSET': 20,
    'BLENDSHAPE_Y_START': 30,
    'BLENDSHAPE_TEXT_FORMAT': '{} ({:.2f})',
    
    # Face Mesh Drawing Colors
    'LABEL_BG_COLOR': (255, 255, 255),
    
    # Camera Error Message
    'CAMERA_ERROR_MSG': 'ERROR: Unable to read from webcam. Please verify your webcam settings.'
}

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================

# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

# ---- State (globals) ----
EYE_CLOSED_START = None
YAWN_START = None
PERCLOS_WIN = deque()
BLINK_TIMES = deque()
EYE_CLOSURE_EVENTS = deque()
EYE_PARTIAL_CLOSURE_START = None

# Event counters
YAWN_COUNT = 0
DROWSY_COUNT = 0
MICROSLEEP_COUNT = 0
YAWN_COUNTED = False
MICROSLEEP_COUNTED = False
DROWSY_COUNTED = False

# Scroll variables
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

        # --- PERCLOS window maintenance ---
        is_closed = 1 if eye_closed_score > CONFIG['EYE_CLOSED_THRESH'] else 0
        PERCLOS_WIN.append((now, is_closed))
        while PERCLOS_WIN and (now - PERCLOS_WIN[0][0]) > CONFIG['PERCLOS_WIN_SEC']:
            PERCLOS_WIN.popleft()
        perclos = (sum(v for _, v in PERCLOS_WIN) / len(PERCLOS_WIN)) if PERCLOS_WIN else 0.0

        # --- Eye closure frequency tracking ---
        global EYE_CLOSURE_EVENTS, EYE_PARTIAL_CLOSURE_START
        
        if eye_closed_score > CONFIG['EYE_PARTIAL_THRESH']:
            if EYE_PARTIAL_CLOSURE_START is None:
                EYE_PARTIAL_CLOSURE_START = now
        else:
            if EYE_PARTIAL_CLOSURE_START is not None:
                closure_duration = now - EYE_PARTIAL_CLOSURE_START
                
                if closure_duration >= CONFIG['MIN_CLOSURE_DURATION']:
                    if not EYE_CLOSURE_EVENTS or (EYE_PARTIAL_CLOSURE_START - EYE_CLOSURE_EVENTS[-1]) > CONFIG['CLOSURE_DEBOUNCE_TIME']:
                        EYE_CLOSURE_EVENTS.append(now)
                
                EYE_PARTIAL_CLOSURE_START = None
        
        while EYE_CLOSURE_EVENTS and (now - EYE_CLOSURE_EVENTS[0]) > CONFIG['EYE_CLOSURE_FREQ_WIN']:
            EYE_CLOSURE_EVENTS.popleft()
        
        frequent_closures = len(EYE_CLOSURE_EVENTS) > CONFIG['EYE_CLOSURE_FREQ_THRESH']

        # --- Microsleep & blink counting ---
        global EYE_CLOSED_START, MICROSLEEP_COUNT, MICROSLEEP_COUNTED
        if is_closed:
            if EYE_CLOSED_START is None:
                EYE_CLOSED_START = now
        else:
            if EYE_CLOSED_START is not None:
                duration = now - EYE_CLOSED_START
                if duration < CONFIG['BLINK_MAX_DURATION']:
                    BLINK_TIMES.append(now)
                EYE_CLOSED_START = None
                MICROSLEEP_COUNTED = False

        while BLINK_TIMES and (now - BLINK_TIMES[0]) > 60.0:
            BLINK_TIMES.popleft()
        blinks_per_min = len(BLINK_TIMES)

        microsleep = (EYE_CLOSED_START is not None) and ((now - EYE_CLOSED_START) >= CONFIG['MICROSLEEP_SEC'])
        
        if microsleep and not MICROSLEEP_COUNTED:
            MICROSLEEP_COUNT += 1
            MICROSLEEP_COUNTED = True

        # --- Yawn detection ---
        global YAWN_START, YAWN_COUNT, YAWN_COUNTED
        yawning = False
        if mouth_lower_down > CONFIG['YAWN_THRESH']:
            if YAWN_START is None:
                YAWN_START = now
            elif (now - YAWN_START) >= CONFIG['YAWN_MIN_SEC']:
                yawning = True
                if not YAWN_COUNTED:
                    YAWN_COUNT += 1
                    YAWN_COUNTED = True
        else:
            YAWN_START = None
            YAWN_COUNTED = False

        # --- Drowsiness decision ---
        global DROWSY_COUNT, DROWSY_COUNTED
        drowsy = microsleep or (perclos >= CONFIG['PERCLOS_DROWSY']) or yawning or frequent_closures
        
        if drowsy and not DROWSY_COUNTED:
            DROWSY_COUNT += 1
            DROWSY_COUNTED = True
        elif not drowsy:
            DROWSY_COUNTED = False

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
    """Continuously run inference on images acquired from the camera."""

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    global SCROLL_OFFSET, MAX_SCROLL

    def mouse_callback(event, x, y, flags, param):
        global SCROLL_OFFSET, MAX_SCROLL
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                SCROLL_OFFSET = max(0, SCROLL_OFFSET - CONFIG['SCROLL_STEP'])
            else:
                SCROLL_OFFSET = min(MAX_SCROLL, SCROLL_OFFSET + CONFIG['SCROLL_STEP'])

    cv2.namedWindow(CONFIG['WINDOW_NAME'])
    cv2.setMouseCallback(CONFIG['WINDOW_NAME'], mouse_callback)

    def save_result(result: vision.FaceLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        if COUNTER % CONFIG['FPS_AVG_FRAME_COUNT'] == 0:
            FPS = CONFIG['FPS_AVG_FRAME_COUNT'] / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

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

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(CONFIG['CAMERA_ERROR_MSG'])

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show FPS
        if CONFIG['SHOW_FPS']:
            fps_text = CONFIG['FPS_TEXT_FORMAT'].format(FPS)
            text_location = (CONFIG['LEFT_MARGIN'], CONFIG['ROW_SIZE'] + CONFIG['FPS_Y_OFFSET'])
            current_frame = image
            cv2.putText(current_frame, fps_text, text_location,
                        CONFIG['FPS_FONT'], CONFIG['FPS_FONT_SIZE'], 
                        CONFIG['FPS_COLOR'], CONFIG['FPS_FONT_THICKNESS'], cv2.LINE_AA)
        else:
            current_frame = image

        if DETECTION_RESULT:
            # Draw landmarks
            if CONFIG['SHOW_FACE_MESH']:
                for face_landmarks in DETECTION_RESULT.face_landmarks:
                    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    face_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                        for landmark in face_landmarks
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

        # Expand right side for blendshapes
        if CONFIG['SHOW_BLENDSHAPES']:
            current_frame = cv2.copyMakeBorder(current_frame, 0, 0, 0,
                                               CONFIG['LABEL_PADDING_WIDTH'],
                                               cv2.BORDER_CONSTANT, None,
                                               CONFIG['LABEL_BG_COLOR'])

        if DETECTION_RESULT:
            legend_x = current_frame.shape[1] - CONFIG['LABEL_PADDING_WIDTH'] + CONFIG['BLENDSHAPE_X_OFFSET']
            legend_y = CONFIG['BLENDSHAPE_Y_START'] - SCROLL_OFFSET
            bar_max_width = CONFIG['LABEL_PADDING_WIDTH'] - 40

            face_blendshapes = DETECTION_RESULT.face_blendshapes

            if face_blendshapes:
                behavior_data = detect_driver_behavior(face_blendshapes, current_frame.shape[0], current_frame)
                
                if behavior_data:
                    # Draw metrics background
                    if CONFIG['SHOW_METRICS']:
                        metrics_x = CONFIG['LEFT_MARGIN'] - CONFIG['METRICS_PADDING']
                        metrics_y = CONFIG['ROW_SIZE'] + CONFIG['METRICS_Y_OFFSET']
                        
                        overlay = current_frame.copy()
                        corner_radius = CONFIG['METRICS_CORNER_RADIUS']
                        
                        # Draw rounded rectangle
                        cv2.rectangle(overlay,
                                    (metrics_x + corner_radius, metrics_y),
                                    (metrics_x + CONFIG['METRICS_WIDTH'] - corner_radius, 
                                     metrics_y + CONFIG['METRICS_HEIGHT']),
                                    CONFIG['METRICS_BG_COLOR'], -1)
                        cv2.rectangle(overlay,
                                    (metrics_x, metrics_y + corner_radius),
                                    (metrics_x + CONFIG['METRICS_WIDTH'], 
                                     metrics_y + CONFIG['METRICS_HEIGHT'] - corner_radius),
                                    CONFIG['METRICS_BG_COLOR'], -1)
                        
                        # Corner circles
                        for dx, dy in [(corner_radius, corner_radius),
                                       (CONFIG['METRICS_WIDTH'] - corner_radius, corner_radius),
                                       (corner_radius, CONFIG['METRICS_HEIGHT'] - corner_radius),
                                       (CONFIG['METRICS_WIDTH'] - corner_radius, CONFIG['METRICS_HEIGHT'] - corner_radius)]:
                            cv2.circle(overlay, (metrics_x + dx, metrics_y + dy),
                                     corner_radius, CONFIG['METRICS_BG_COLOR'], -1)
                        
                        cv2.addWeighted(overlay, CONFIG['METRICS_BG_OPACITY'], 
                                      current_frame, 1 - CONFIG['METRICS_BG_OPACITY'], 0, current_frame)
                        
                        # Display metrics
                        metrics_data = [
                            (CONFIG['LABEL_PERCLOS'].format(behavior_data['perclos']), CONFIG['PERCLOS_Y_OFFSET']),
                            (CONFIG['LABEL_BLINKS'].format(behavior_data['blinks_per_min']), CONFIG['BLINKS_Y_OFFSET']),
                            (CONFIG['LABEL_CLOSURES'].format(behavior_data['closure_count']), CONFIG['CLOSURES_Y_OFFSET']),
                            (CONFIG['LABEL_YAWNS'].format(behavior_data['yawn_count']), CONFIG['YAWNS_Y_OFFSET']),
                            (CONFIG['LABEL_MICROSLEEPS'].format(behavior_data['microsleep_count']), CONFIG['MICROSLEEPS_Y_OFFSET']),
                            (CONFIG['LABEL_DROWSY_EVENTS'].format(behavior_data['drowsy_count']), CONFIG['DROWSY_EVENTS_Y_OFFSET']),
                        ]
                        
                        for text, y_offset in metrics_data:
                            cv2.putText(current_frame, text,
                                       (CONFIG['LEFT_MARGIN'], CONFIG['ROW_SIZE'] + y_offset),
                                       CONFIG['METRICS_FONT'], CONFIG['METRICS_FONT_SIZE'],
                                       CONFIG['METRICS_TEXT_COLOR'], CONFIG['METRICS_FONT_THICKNESS'], cv2.LINE_AA)
                    
                    # Display warnings
                    if CONFIG['SHOW_WARNINGS']:
                        frame_width = current_frame.shape[1] - CONFIG['LABEL_PADDING_WIDTH']
                        
                        warning_checks = [
                            (behavior_data['microsleep'], CONFIG['WARNING_MICROSLEEP'], 
                             CONFIG['CONSOLE_MICROSLEEP'].format(behavior_data['microsleep_count'])),
                            (behavior_data['yawning'], CONFIG['WARNING_YAWNING'],
                             CONFIG['CONSOLE_YAWN'].format(behavior_data['yawn_count'])),
                            (behavior_data['frequent_closures'], CONFIG['WARNING_FREQUENT_CLOSURES'],
                             CONFIG['CONSOLE_FREQUENT_CLOSURES']),
                            (behavior_data['drowsy'], CONFIG['WARNING_DROWSY'],
                             CONFIG['CONSOLE_DROWSY'].format(behavior_data['drowsy_count'])),
                        ]
                        
                        for condition, warning_text, console_msg in warning_checks:
                            if condition:
                                (text_width, _), _ = cv2.getTextSize(warning_text,
                                                                     CONFIG['WARNING_FONT'],
                                                                     CONFIG['WARNING_FONT_SIZE'],
                                                                     CONFIG['WARNING_FONT_THICKNESS'])
                                right_x = frame_width - text_width - CONFIG['WARNING_RIGHT_MARGIN']
                                cv2.putText(current_frame, warning_text,
                                           (right_x, CONFIG['WARNING_Y_POSITION']),
                                           CONFIG['WARNING_FONT'], CONFIG['WARNING_FONT_SIZE'],
                                           CONFIG['WARNING_COLOR'], CONFIG['WARNING_FONT_THICKNESS'], cv2.LINE_AA)
                                print(console_msg)
                                break
                
                # Draw blendshapes
                if CONFIG['SHOW_BLENDSHAPES']:
                    num_blendshapes = len(face_blendshapes[0])
                    total_height = num_blendshapes * (CONFIG['BLENDSHAPE_BAR_HEIGHT'] + CONFIG['BLENDSHAPE_GAP_BETWEEN_BARS'])
                    MAX_SCROLL = max(0, total_height - current_frame.shape[0] + 60)
                    
                    for category in face_blendshapes[0]:
                        if legend_y + CONFIG['BLENDSHAPE_BAR_HEIGHT'] > 0 and legend_y < current_frame.shape[0]:
                            text = CONFIG['BLENDSHAPE_TEXT_FORMAT'].format(category.category_name, round(category.score, 2))
                            (text_width, _), _ = cv2.getTextSize(text, CONFIG['BLENDSHAPE_FONT'],
                                                                CONFIG['BLENDSHAPE_FONT_SIZE'],
                                                                CONFIG['BLENDSHAPE_FONT_THICKNESS'])

                            cv2.putText(current_frame, text,
                                        (legend_x, legend_y + (CONFIG['BLENDSHAPE_BAR_HEIGHT'] // 2) + 5),
                                        CONFIG['BLENDSHAPE_FONT'], CONFIG['BLENDSHAPE_FONT_SIZE'],
                                        CONFIG['BLENDSHAPE_TEXT_COLOR'], CONFIG['BLENDSHAPE_FONT_THICKNESS'], cv2.LINE_AA)

                            bar_width = int(bar_max_width * category.score)
                            cv2.rectangle(current_frame,
                                        (legend_x + text_width + CONFIG['BLENDSHAPE_TEXT_GAP'], legend_y),
                                        (legend_x + text_width + CONFIG['BLENDSHAPE_TEXT_GAP'] + bar_width,
                                         legend_y + CONFIG['BLENDSHAPE_BAR_HEIGHT']),
                                        CONFIG['BLENDSHAPE_BAR_COLOR'], -1)

                        legend_y += (CONFIG['BLENDSHAPE_BAR_HEIGHT'] + CONFIG['BLENDSHAPE_GAP_BETWEEN_BARS'])

        cv2.imshow(CONFIG['WINDOW_NAME'], current_frame)

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