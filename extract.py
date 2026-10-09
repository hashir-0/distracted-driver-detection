import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

dataset_path = "dataset_up"

# Selected landmarks
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
        (a.x-b.x)**2 +
        (a.y-b.y)**2
    )

def angle(a, b, c):

    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(
        c[1]-b[1],
        c[0]-b[0]
    ) - np.arctan2(
        a[1]-b[1],
        a[0]-b[0]
    )

    ang = abs(radians * 180 / np.pi)

    if ang > 180:
        ang = 360 - ang

    return ang

rows = []

for label in os.listdir(dataset_path):

    folder = os.path.join(dataset_path, label)

    if not os.path.isdir(folder):
        continue

    print("Processing", label)

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = pose.process(rgb)

        if not result.pose_landmarks:
            continue

        lm = result.pose_landmarks.landmark

        points = [lm[i] for i in LANDMARKS]

        features = []

        # Pairwise distances
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                features.append(
                    distance(points[i], points[j])
                )

        # Arm angles
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

        features.append(label)

        rows.append(features)

df = pd.DataFrame(rows)

df.to_csv(
    "driver_pose_features.csv",
    index=False
)

print("Samples:", len(df))