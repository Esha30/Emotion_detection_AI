import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import io

# Page config
st.set_page_config(page_title="Emotion Detection AI", page_icon="😊", layout="wide")

# Premium CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0d14 0%, #0d1117 100%); }
h1 {
    background: linear-gradient(90deg, #00d4ff, #a855f7, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; text-align: center;
    font-size: 3rem !important; font-weight: 800 !important; letter-spacing: -1px;
}
h2, h3 { color: #e2e8f0 !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1f2e, #1e2535) !important;
    border: 1px solid #2a3550 !important; border-radius: 16px !important;
    padding: 20px !important; box-shadow: 0 4px 24px rgba(0,212,255,0.08) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,212,255,0.18) !important;
}
[data-testid="metric-container"] label { color: #8892a4 !important; font-size: 13px !important; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: #a855f7 !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117, #111827) !important;
    border-right: 1px solid #1e2535 !important;
}
.history-item {
    background: #1a1f2e; border-left: 3px solid #00d4ff;
    border-radius: 8px; padding: 8px 14px; margin: 5px 0;
    font-size: 13px; color: #cbd5e1;
}
.stDownloadButton button {
    background: linear-gradient(135deg, #00d4ff, #a855f7) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
}
.stButton button {
    background: #1e2535 !important; color: #e2e8f0 !important;
    border: 1px solid #2a3550 !important; border-radius: 10px !important;
}
hr { border-color: #1e2535 !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>😊 Emotion Detection AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#8892a4;font-size:16px;margin-bottom:1.5rem;'>"
    "Upload a face image or use your camera — AI detects emotions instantly with full confidence scores</p>",
    unsafe_allow_html=True
)
st.divider()

# Load model (cached)
@st.cache_resource
def load_emotion_model():
    return load_model('model/emotion_model.keras', compile=False)

model = load_emotion_model()

CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOJIS = {'Angry':'😠','Disgust':'🤢','Fear':'😨','Happy':'😄','Neutral':'😐','Sad':'😢','Surprise':'😲'}
COLORS = {'Angry':'#ff4b4b','Disgust':'#8bc34a','Fear':'#9c27b0','Happy':'#ffd700','Neutral':'#00bcd4','Sad':'#2196f3','Surprise':'#ff9800'}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Session state
if 'history' not in st.session_state:
    st.session_state.history = []

def detect_faces(gray):
    for scale, neigh, min_s in [(1.1, 4, 30), (1.05, 2, 20), (1.03, 1, 15)]:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neigh, minSize=(min_s, min_s))
        if len(faces) > 0:
            return faces
    return np.array([])

def predict_emotion(image):
    try:
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = detect_faces(cv2.equalizeHist(gray))
        fallback = len(faces) == 0
        if fallback:
            h, w = gray.shape
            faces = np.array([[0, 0, w, h]])

        results, all_probs = [], []
        for (x, y, w, h) in faces:
            face_in = np.expand_dims(np.expand_dims(
                cv2.resize(gray[y:y+h, x:x+w], (48, 48)).astype('float32') / 255.0, -1), 0)
            pred = model.predict(face_in, verbose=0)[0]
            label = CLASSES[np.argmax(pred)]
            confidence = float(np.max(pred)) * 100
            probs = {cls: round(float(p)*100, 2) for cls, p in zip(CLASSES, pred)}
            if not fallback:
                hx = COLORS.get(label, '#00dc64').lstrip('#')
                r, g, b = tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))
                cv2.rectangle(img_array, (x,y), (x+w,y+h), (b,g,r), 2)
                cv2.putText(img_array, f"{label} ({confidence:.1f}%)", (x, max(y-10,10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (b,g,r), 2)
            results.append((label, confidence))
            all_probs.append(probs)
        return results, img_array, fallback, all_probs
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return [], np.array(image.convert('RGB')), False, []

# Sidebar
with st.sidebar:
    st.markdown("## 🕐 Detection History")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-10:]):
            st.markdown(f"<div class='history-item'>{item}</div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("<p style='color:#555;font-size:13px;'>No detections yet.</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 🎭 Emotion Guide")
    for emotion, emoji in EMOJIS.items():
        color = COLORS[emotion]
        st.markdown(f"<span style='color:{color};font-size:18px;'>{emoji}</span> &nbsp; **{emotion}**", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='color:#374151;font-size:11px;text-align:center;'>Powered by TensorFlow · OpenCV · Streamlit</p>", unsafe_allow_html=True)

# Input section
col1, col2 = st.columns(2)
with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a face image", type=["jpg","jpeg","png"])
with col2:
    st.subheader("📷 Or Use Camera")
    camera_photo = st.camera_input("Take a photo")

input_image = None
if uploaded_file:
    try:
        input_image = Image.open(uploaded_file)
    except Exception:
        st.error("❌ Could not open image. Please upload a valid JPG or PNG file.")
elif camera_photo:
    input_image = Image.open(camera_photo)

# Results
if input_image:
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**📥 Input Image**")
        st.image(input_image, width='stretch')

    with st.spinner("🔍 Analyzing emotion with AI..."):
        results, output_img, fallback, all_probs = predict_emotion(input_image)

    with col_b:
        title = "🎯 Detected Emotion" if not fallback else "🎯 Detected Emotion (full image)"
        st.markdown(f"**{title}**")
        st.image(output_img, width='stretch')

    if results:
        st.divider()
        if fallback:
            st.info("ℹ️ Face not clearly detected — the full image was analyzed for emotion.")

        st.subheader("📊 Detection Results")
        metric_cols = st.columns(len(results))
        for i, (label, confidence) in enumerate(results):
            with metric_cols[i]:
                st.metric(label=f"Face {i+1}", value=f"{EMOJIS.get(label,'')} {label}", delta=f"{confidence:.1f}% confident")

        # Confidence charts
        if all_probs:
            st.divider()
            st.subheader("📈 Full Confidence Distribution")
            chart_cols = st.columns(len(all_probs))
            for i, probs in enumerate(all_probs):
                with chart_cols[i]:
                    if len(all_probs) > 1:
                        st.markdown(f"**Face {i+1}**")
                    df = pd.DataFrame({'Emotion': [f"{EMOJIS[e]} {e}" for e in probs], 'Confidence (%)': list(probs.values())}).set_index('Emotion')
                    st.bar_chart(df, color="#00d4ff")
                    for emotion, pct in sorted(probs.items(), key=lambda x: -x[1]):
                        color = COLORS[emotion]
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;font-size:13px;color:#cbd5e1;margin:2px 0;'>"
                            f"<span>{EMOJIS[emotion]} {emotion}</span><span style='color:{color};'><b>{pct:.1f}%</b></span></div>",
                            unsafe_allow_html=True)
                        st.progress(pct / 100)

        # Download
        st.divider()
        buf = io.BytesIO()
        Image.fromarray(output_img).save(buf, format="PNG")
        st.download_button(label="⬇️ Download Result Image", data=buf.getvalue(), file_name="emotion_result.png", mime="image/png")

        # History
        for label, confidence in results:
            entry = f"{EMOJIS.get(label,'')} {label} — {confidence:.1f}%"
            if not st.session_state.history or st.session_state.history[-1] != entry:
                st.session_state.history.append(entry)

        st.success("✅ Detection complete!")
else:
    st.info("👆 Upload an image or take a photo to get started.")

st.divider()
st.markdown("<p style='text-align:center;color:#374151;font-size:12px;'>© 2025 Emotion Detection AI · Built with TensorFlow, OpenCV & Streamlit</p>", unsafe_allow_html=True)
