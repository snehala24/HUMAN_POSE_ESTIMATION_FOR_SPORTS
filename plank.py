import cv2
import mediapipe as mp
import argparse
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', type=str, required=True, help='Path to the input video (e.g., Plank.mp4)')
parser.add_argument('-o', '--output', type=str, default='Plank_output.mp4', help='Path to save the output video (e.g., Plank_output.mp4)')

args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

# Plank counter variables
plank_counter = 0
correct_plank_counter = 0
incorrect_plank_counter = 0
position = None  # None, 'high', 'low'
display_pos = 'None'

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for angle calculation
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angles
            left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
            right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

            # Plank counter logic
            if left_shoulder_angle < 30 and right_shoulder_angle < 30:
                position = "high"
                display_pos = "HIGH"
            if left_hip_angle < 30 and right_hip_angle < 30 and position == 'high':
                position = "low"
                display_pos = "LOW"
                plank_counter += 1
                # Define criteria for correct plank
                if 0 < left_shoulder_angle < 30 and 0 < right_shoulder_angle < 30:
                    correct_plank_counter += 1
                else:
                    incorrect_plank_counter += 1

            # Calculate accuracy
            total_planks = correct_plank_counter + incorrect_plank_counter
            if total_planks > 0:
                accuracy = (correct_plank_counter / total_planks) * 100
            else:
                accuracy = 0

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw text
            cv2.putText(image, f"PLANKS: {plank_counter}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(image, f"CORRECTly DONE PLANKS {correct_plank_counter}", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # Display angles at top right corner
            text_y_position = 50
            for angle_name, angle_value in [
                ("Left Shoulder Angle", left_shoulder_angle),
                ("Right Shoulder Angle", right_shoulder_angle),
                ("Left Hip Angle", left_hip_angle),
                ("Right Hip Angle", right_hip_angle)
            ]:
                cv2.putText(image, f"{angle_name}: {angle_value:.2f}", 
                            (width - 200, text_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                text_y_position += 30

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
