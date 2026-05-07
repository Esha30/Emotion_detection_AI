import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Page config
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="😊",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #00d4ff; text-align: center; }
    .stMetric { background: #1e2530; border-radius: 10px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("😊 Emotion Detection AI")
st.markdown("<p style='text-align:center; color:#aaa;'>Upload a face image — AI will detect the emotion instantly!</p>", unsafe_allow_html=True)
st.divider()

# Load model (cached so it only loads once)
@st.cache_resource
def load_emotion_model():
    return load_model('model/emotion_model.keras', compile=False)

model = load_emotion_model()

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_emojis = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😄', 'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces(gray):
    """Try multiple parameter sets to detect faces robustly."""
    # Attempt 1: standard
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    if len(faces) > 0:
        return faces

    # Attempt 2: more relaxed
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20)
    )
    if len(faces) > 0:
        return faces

    # Attempt 3: very relaxed
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.03, minNeighbors=1, minSize=(15, 15)
    )
    return faces

def predict_emotion(image):
    # Convert to RGB numpy array
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Enhance contrast for better detection
    gray_eq = cv2.equalizeHist(gray)

    faces = detect_faces(gray_eq)
    fallback = False

    if len(faces) == 0:
        # Fallback: treat the entire image as a face
        h, w = gray.shape
        faces = np.array([[0, 0, w, h]])
        fallback = True

    results = []
    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_gray, (48, 48)).astype('float32') / 255.0
        face_input = np.expand_dims(np.expand_dims(face_resized, -1), 0)
        prediction = model.predict(face_input, verbose=0)
        label = classes[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        if not fallback:
            # Draw bounding box and label on image
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 220, 100), 2)
            cv2.putText(
                img_array, f"{label} ({confidence:.1f}%)",
                (x, max(y - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 100), 2
            )
        results.append((label, confidence))

    return results, img_array, fallback

# --- Upload Section ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("📷 Or Use Camera")
    camera_photo = st.camera_input("Take a photo")

# Use whichever input was provided
input_image = None
if uploaded_file:
    input_image = Image.open(uploaded_file)
elif camera_photo:
    input_image = Image.open(camera_photo)

if input_image:
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.image(input_image, caption="Input Image", use_container_width=True)

    with st.spinner("🔍 Analyzing emotion..."):
        results, output_img, fallback = predict_emotion(input_image)

    with col_b:
        caption = "Detected Emotion" if not fallback else "Detected Emotion (full image used)"
        st.image(output_img, caption=caption, use_container_width=True)

    if results:
        st.divider()
        if fallback:
            st.info("ℹ️ Face not clearly detected — analyzed full image for emotion.")
        st.subheader("📊 Results")
        cols = st.columns(len(results))
        for i, (label, confidence) in enumerate(results):
            with cols[i]:
                emoji = emotion_emojis.get(label, "")
                st.metric(
                    label=f"Face {i+1}",
                    value=f"{emoji} {label}",
                    delta=f"{confidence:.1f}% confident"
                )
        st.success("✅ Detection complete!")
else:
    st.info("👆 Upload an image or use your camera to get started.")

st.divider()
st.markdown("<p style='text-align:center; color:#555; font-size:12px;'>Built with TensorFlow, OpenCV & Streamlit</p>", unsafe_allow_html=True)
