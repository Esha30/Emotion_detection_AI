# 😊 Emotion Detection AI

A real-time facial emotion detection web app built with **TensorFlow**, **OpenCV**, and **Streamlit**. Upload a face image or use your webcam — the AI instantly detects emotions with full confidence scores.

---

## 🚀 Features

- 🖼️ **Image Upload** — JPG, PNG, JPEG support
- 📷 **Live Camera Input** — Take a photo directly in the browser
- 🤖 **AI Emotion Detection** — Detects 7 emotions with bounding boxes
- 📊 **Confidence Bar Charts** — Visual breakdown of all emotion probabilities
- 📈 **Progress Bars** — Per-emotion confidence display
- 🕐 **Detection History** — Sidebar tracks your last 10 results
- ⬇️ **Download Result** — Save the annotated output image
- 🌙 **Premium Dark UI** — Glassmorphism-inspired design with gradient accents
- ✅ **Error Handling** — Graceful fallback for unclear face images

---

## 🎭 Supported Emotions

| Emotion   | Emoji |
|-----------|-------|
| Angry     | 😠    |
| Disgust   | 🤢    |
| Fear      | 😨    |
| Happy     | 😄    |
| Neutral   | 😐    |
| Sad       | 😢    |
| Surprise  | 😲    |

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Esha30/Emotion_detection_AI.git
cd Emotion_detection_AI
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 📁 Project Structure

```
Emotion_detection_AI/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── model/
│   └── emotion_model.keras # Trained FER model (TensorFlow/Keras)
├── FER Code.ipynb          # Model training notebook
├── webcam_emotion.py       # Standalone webcam script (OpenCV)
└── README.md
```

---

## 🧠 Model Details

- **Architecture**: CNN trained on the FER-2013 dataset
- **Input**: 48×48 grayscale face images
- **Output**: 7-class softmax probabilities
- **Framework**: TensorFlow / Keras
- **Face Detection**: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)

---

## 📦 Dependencies

```
streamlit
tensorflow
opencv-python-headless
Pillow
pandas
```

---

## 👩‍💻 Author

**Esha** — [GitHub](https://github.com/Esha30)

---

## 📄 License

MIT License — free to use and modify.