import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_video(video_path, exercise_type):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if exercise_type == "pushup":
        min_detection_confidence = 0.5
        min_tracking_confidence = 0.5

        with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                          min_tracking_confidence=min_tracking_confidence) as pose:
            pushup_counter = 0
            position = None  # None, 'up', 'down'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                    if left_elbow_angle > 160:
                        position = "up"
                    elif left_elbow_angle < 90:
                        if position == "up":
                            position = "down"
                            pushup_counter += 1

                    cv2.putText(image, f"PUSHUPS: {pushup_counter}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.putText(image, f"POSITION: {position}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                except Exception as e:
                    print(e)
                    continue

                out.write(image)
                cv2.imshow('Pose Detection', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()

    elif exercise_type == "squat":
        min_detection_confidence = 0.5
        min_tracking_confidence = 0.5

        with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                          min_tracking_confidence=min_tracking_confidence) as pose:
            squat_counter = 0
            position = None  # None, 'up', 'down'

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark

                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                    left_knee_angle = calculate_angle(left_hip, left_knee, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x)
                    right_knee_angle = calculate_angle(right_hip, right_knee, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)

                    if left_knee_angle > 160 and right_knee_angle > 160:
                        if position != "up":
                            position = "up"
                    elif left_knee_angle < 100 and right_knee_angle < 100:
                        if position == "up":
                            position = "down"
                            squat_counter += 1

                    cv2.putText(image, f"SQUATS: {squat_counter}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.putText(image, f"POSITION: {position}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

                except Exception as e:
                    print(e)
                    continue

                out.write(image)
                cv2.imshow('Pose Detection', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()

    return out_path

def main():
    st.title('Exercise Detection')

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    exercise_type = st.selectbox("Select the type of exercise", ["pushup", "squat"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.video(uploaded_file)

        if st.button("Process Video"):
            st.write("Processing video...")
            out_path = process_video(tfile.name, exercise_type)
            st.write("Video processed successfully.")
            st.video(out_path)

if __name__ == '__main__':
    main()
