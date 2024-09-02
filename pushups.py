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
parser.add_argument('-v', '--video', type=str, required=True, help='Path to the input video (e.g., Pushup.mp4)')
parser.add_argument('-o', '--output', type=str, default='Pushup_output.mp4', help='Path to save the output video (e.g., Pushup_output.mp4)')

args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

# Curl counter variables
push_up_counter = 0
correct_push_up_counter = 0
incorrect_push_up_counter = 0
position = None  # None, 'up', 'down'
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
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

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
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # Push-up counter logic
            if left_elbow_angle > 160 and right_elbow_angle > 160:
                position = "up"
                display_pos = "UP"
            if left_elbow_angle < 90 and right_elbow_angle < 90 and position == 'up':
                position = "down"
                display_pos = "DOWN"
                push_up_counter += 1
                # Define criteria for correct rep
                if 70 < left_elbow_angle < 90 and 70 < right_elbow_angle < 90:
                    correct_push_up_counter += 1
                else:
                    incorrect_push_up_counter += 1

            # Calculate accuracy
            total_push_ups = correct_push_up_counter + incorrect_push_up_counter
            if total_push_ups > 0:
                accuracy = (correct_push_up_counter / total_push_ups) * 100
            else:
                accuracy = 0

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw text
            cv2.putText(image, f"REPS: {push_up_counter}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(image, f"POSITION: {display_pos}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(image, f"CORRECT REPS: {correct_push_up_counter}", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.putText(image, f"INCORRECT REPS: {incorrect_push_up_counter}", (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(image, f"ACCURACY: {accuracy:.2f}%", (50, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 165, 0), 2)

            # Display angles at top right corner
            text_y_position = 50
            for angle_name, angle_value in [
                ("Left Elbow", left_elbow_angle),
                ("Right Elbow", right_elbow_angle),
                ("Left Knee", left_knee_angle),
                ("Right Knee", right_knee_angle)
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
