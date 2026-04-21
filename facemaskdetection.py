import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ===============================
# PARAMETERS
# ===============================
IMG_SIZE = 100
EPOCHS = 10
MODEL_NAME = "mask_detector.model"
DATASET_DIR = "dataset"

# ===============================
# TRAIN MODEL
# ===============================
def train_model():
    print("[INFO] Training model...")

    data = []
    labels = []
    categories = ["with_mask", "without_mask"]

    for category in categories:
        path = os.path.join(DATASET_DIR, category)
        label = categories.index(category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(image)
            labels.append(label)

    data = np.array(data, dtype="float32") / 255.0
    labels = to_categorical(labels, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test))
    model.save(MODEL_NAME)
    print("[INFO] Model saved successfully!")

# ===============================
# LOAD OR TRAIN MODEL
# ===============================
if not os.path.exists(MODEL_NAME):
    train_model()

model = load_model(MODEL_NAME)

# ===============================
# LOAD FACE DETECTOR
# ===============================
face_net = cv2.dnn.readNet(
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel",
    "face_detector/deploy.prototxt"
)

# ===============================
# START WEBCAM
# ===============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("[ERROR] Webcam not accessible")
    exit()

print("[INFO] Starting real-time detection... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300,300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1, y1, x2, y2 = box.astype(int)

            # SAFE BOUNDING BOX
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 3))

            prediction = model.predict(face, verbose=0)[0]
            label = "Mask" if prediction[0] > prediction[1] else "No Mask"
            color = (0,255,0) if label == "Mask" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()