import cap
import cv2

def extract_keyframes(video_path, output_path='summary.mp4', num_frames=100):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)

    # Create VideoWriter object to save the summary video
    summary_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Check if the VideoWriter is successfully created
    if not summary_writer.isOpened():
        print(f"Error: Could not create VideoWriter for '{output_path}'")
        cap.release()
        return

    # Extract key frames evenly spaced throughout the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Corrected calculation for frame indices
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    for frame_index in frame_indices:
        cap.set(1, frame_index)  # Set the frame index
        ret, frame = cap.read()
        if ret:
            summary_writer.write(frame)

    # Release video capture and writer objects
    cap.release()
    summary_writer.release()
