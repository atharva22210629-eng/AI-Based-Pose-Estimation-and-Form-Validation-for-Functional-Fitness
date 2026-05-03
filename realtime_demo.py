import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Angle calculation
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

cap = cv2.VideoCapture(0)

counter = 0
stage = None
start_time = time.time()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get key points
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)

            # FORM LOGIC
            if angle < 90:
                status = "GOOD FORM"
                color = (0, 255, 0)
                stage = "down"

            elif angle < 120:
                status = "GO LOWER"
                color = (0, 255, 255)

            else:
                status = "POOR FORM"
                color = (0, 0, 255)

            # REP COUNTER
            if angle > 140 and stage == "down":
                stage = "up"
                counter += 1

            # FEEDBACK
            feedback = ""

            if angle > 120:
                feedback = "Go Lower"

            # Back posture
            if shoulder[0] < hip[0] - 0.1:
                feedback = "Keep Chest Up"

            # Knee alignment
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

            if abs(left_knee.x - right_knee.x) < 0.05:
                feedback = "Keep Knees Out"

            # Timer
            elapsed = int(time.time() - start_time)

            # Display angle near knee
            cv2.putText(image, f"{int(angle)} deg",
                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # TEXT OVERLAY (no background)
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(image, "HYROX FORM COACH", (20,30), font, 0.7, (0,255,255), 2)
            cv2.putText(image, f"Reps: {counter}", (20,60), font, 0.7, (255,255,255), 2)
            cv2.putText(image, f"Time: {elapsed}s", (20,90), font, 0.7, (255,255,255), 2)
            cv2.putText(image, f"Status: {status}", (20,120), font, 0.7, color, 2)
            cv2.putText(image, f"Feedback: {feedback}", (20,150), font, 0.7, (0,200,255), 2)

            # Draw skeleton
            mp_draw.draw_landmarks(image, results.pose_landmarks,
                                   mp_pose.POSE_CONNECTIONS)

        cv2.imshow("HYROX AI COACH", image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()