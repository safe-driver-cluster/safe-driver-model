import cv2
import mediapipe as mp
import numpy as np
import time
import threading

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Eye and mouth landmark indices (MediaPipe)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402]

# Thresholds
EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.65
DROWSY_FRAMES = 20
YAWN_FRAMES = 15

# State variables
closed_frames = 0
yawn_frames = 0
blink_counter = 0
is_drowsy = False
is_yawning = False

def euclidean_dist(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_ear(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = euclidean_dist(mouth[1], mouth[2])
    B = euclidean_dist(mouth[0], mouth[3])
    return A / B

def get_gps_data():
    # Simulated GPS data for Windows
    return {
        "lat": 6.9271,
        "lon": 79.8612,
        "speed": 42.5
    }

def log_gps_data():
    while True:
        gps = get_gps_data()
        print(f"[GPS] Lat: {gps['lat']} | Lon: {gps['lon']} | Speed: {gps['speed']} km/h")
        time.sleep(5)

def monitor_driver():
    global closed_frames, blink_counter, yawn_frames, is_drowsy, is_yawning

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = [(int(p.x * w), int(p.y * h)) for p in results.multi_face_landmarks[0].landmark]

            left_eye = [landmarks[i] for i in LEFT_EYE]
            right_eye = [landmarks[i] for i in RIGHT_EYE]
            mouth = [landmarks[i] for i in MOUTH]

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)

            # Drowsiness detection
            if ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                if closed_frames > 3:
                    blink_counter += 1
                closed_frames = 0

            is_drowsy = closed_frames >= DROWSY_FRAMES

            # Yawning detection
            if mar > MAR_THRESHOLD:
                yawn_frames += 1
            else:
                yawn_frames = 0

            is_yawning = yawn_frames >= YAWN_FRAMES

            # Draw text
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {blink_counter}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if is_drowsy:
                cv2.putText(frame, "DROWSY!", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            if is_yawning:
                cv2.putText(frame, "YAWNING!", (220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gps_thread = threading.Thread(target=log_gps_data)
    gps_thread.daemon = True
    gps_thread.start()

    monitor_driver()
