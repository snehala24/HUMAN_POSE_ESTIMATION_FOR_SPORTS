
---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Libraries:** 
  - [OpenCV](https://opencv.org/) – Real-time computer vision
  - [MediaPipe](https://mediapipe.dev/) – Human pose estimation
  - [Streamlit](https://streamlit.io/) – Interactive frontend for user interface

---

## 🧠 How it Works

1. **MediaPipe Pose Estimation** is used to extract 33 key landmarks from the human body.
2. **OpenCV** captures the live webcam feed and displays annotated frames.
3. **Custom logic** in `plank.py` and `pushups.py` measures:
   - Elbow, knee, shoulder, and hip angles
   - Repetition counts based on movement patterns
4. `stream.py` offers a visual interface to run pose detection with simple UI.

---

## 🏃‍♀️ Use Cases

- Fitness tracking and automated workout rep counting
- Gym form correction using pose analytics
- Biomechanical movement analysis
- Interactive fitness applications

---

## ▶️ Getting Started

### 🔧 Requirements

```bash
pip install opencv-python mediapipe streamlit

![image](https://github.com/user-attachments/assets/f1284bb5-3032-4143-9952-b8934b67024c)
