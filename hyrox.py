import cv2, mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_pose.Pose() as pose:
    while True:
        ok, frame = cap.read()
        if not ok: break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if res.pose_landmarks:
            mp_draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("Try this", img)
        if cv2.waitKey(1) & 0xFF == 27: break
cap.release(); cv2.destroyAllWindows()
