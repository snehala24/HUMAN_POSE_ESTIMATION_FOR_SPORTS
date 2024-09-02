# app.py
import streamlit as st
import os
import subprocess

# Title of the web app
st.title('3 Exercise Monitoring System')

# Function to process video based on exercise selected
def process_video(video_file, exercise):
    # Ensure temp_videos directory exists
    os.makedirs("temp_videos", exist_ok=True)

    if video_file is not None:
        # Save the uploaded video locally
        with open(os.path.join("temp_videos", "uploaded_video.mp4"), "wb") as f:
            f.write(video_file.read())

        # Define paths for input and output videos
        input_video_path = os.path.join("temp_videos", "uploaded_video.mp4")
        output_video_path = os.path.join("temp_videos", f"{exercise}_output.mp4")

        # Run the appropriate script based on exercise selected
        if exercise == 'Pushups':
            command = f"python pushups.py -v {input_video_path} -o {output_video_path}"
        elif exercise == 'Squats':
            command = f"python gym_code.py -v {input_video_path} -o {output_video_path}"
        elif exercise == 'Plank':
            command = f"python plank.py -v {input_video_path} -o {output_video_path}"

        # Execute the command to process the video
        try:
            subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            st.success(f"Processing video for {exercise} exercise complete!")

            # Display the processed output video
            st.video(output_video_path)

        except subprocess.CalledProcessError as e:
            st.error(f"Error processing video: {e.stderr}")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

# Dropdown to select exercise
exercise = st.selectbox('Select Exercise', ['Pushups', 'Squats', 'Plank'])

# Process uploaded video and exercise selection
if st.button("Process Video"):
    process_video(uploaded_file, exercise)
