import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Load trained model from models folder (use .keras format for Keras 3)
model = load_model('model/emotion_model.keras', compile=False)  # <- use .keras file

# ✅ Emotion labels (make sure order matches training)
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ✅ Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ✅ Preprocess face
def preprocess_face(face_img):
    face = cv2.resize(face_img, (48, 48))      # resize to 48x48
    face = face.astype('float32') / 255.0      # normalize
    face = np.expand_dims(face, axis=-1)       # (48,48,1)
    face = np.expand_dims(face, axis=0)        # (1,48,48,1)
    return face

# ✅ Predict emotion
def predict_emotion(face_img):
    processed = preprocess_face(face_img)
    prediction = model.predict(processed, verbose=0)
    class_idx = np.argmax(prediction)
    return classes[class_idx]

# ✅ Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]

        # Predict
        label = predict_emotion(face_gray)

        # Draw rectangle + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show output
    cv2.imshow('Emotion Detection', frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()