import cv2
import mediapipe as mp
import numpy as np
import joblib
import math

# Load model, encoder and PCA
model = joblib.load("driver_model.pkl")
encoder = joblib.load("label_encoder.pkl")
pca = joblib.load("pca.pkl")

# Labels
label_names = {
    "c0": "Safe Driving",
    "c1": "Texting Right",
    "c2": "Phone Right",
    "c3": "Texting Left",
    "c4": "Phone Left",
    "c5": "Operating Radio",
    "c6": "Drinking",
    "c7": "Reaching Behind",
    "c8": "Hair/Makeup",
    "c9": "Talking"
}

danger_classes = [
    "c1",
    "c2",
    "c3",
    "c4",
    "c6",
    "c7"
]

# Same landmarks used in extract.py
LANDMARKS = [
    0,   # nose
    7,   # left ear
    8,   # right ear
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24   # right hip
]

def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )

def angle(a, b, c):

    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    ang = abs(radians * 180.0 / np.pi)

    if ang > 180:
        ang = 360 - ang

    return ang

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose.process(rgb)

    if result.pose_landmarks:

        lm = result.pose_landmarks.landmark

        points = [lm[i] for i in LANDMARKS]

        features = []

        # Pairwise distances
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                features.append(
                    distance(points[i], points[j])
                )

        # Angles
        features.append(
            angle(lm[11], lm[13], lm[15])
        )

        features.append(
            angle(lm[12], lm[14], lm[16])
        )

        features.append(
            angle(lm[23], lm[11], lm[13])
        )

        features.append(
            angle(lm[24], lm[12], lm[14])
        )

        features = np.array(features).reshape(1, -1)

        # PCA transform
        features = pca.transform(features)

        # Prediction
        probabilities = model.predict_proba(features)

        prediction = np.argmax(probabilities)

        confidence = np.max(probabilities) * 100

        label = encoder.inverse_transform(
            [prediction]
        )[0]

        if confidence<70:
            display_label="unknown"

        else:    

            display_label = label_names.get(
            label,
            label)

        cv2.putText(
            frame,
            f"{display_label} ({confidence:.1f}%)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if confidence>=70 and label in danger_classes:

            cv2.putText(
                frame,
                "DISTRACTED!",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow(
        "Driver Monitoring System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()