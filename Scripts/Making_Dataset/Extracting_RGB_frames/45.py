import cv2
import os

def extract_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video information
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create the output folder if it doesn't exist
    output_folder_path = os.path.join(output_folder, video_name)
    os.makedirs(output_folder_path, exist_ok=True)

    # Read frames and save them
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_filename = f"{video_name}_frame_{frame_count:04d}.png"
        frame_path = os.path.join(output_folder_path, frame_filename)
        cv2.imwrite(frame_path, frame)

    # Release the video capture object
    cap.release()

    print(f"Frames extracted from {video_name} and saved in {output_folder_path}")

if __name__ == "__main__":
    video_folder = r"D:\Unity\Final\Recordings"
    output_folder = r"D:\Unity\Final\Extracted_Frames"

    # Iterate over each video in the folder
    for video_filename in os.listdir(video_folder):
        if video_filename == "Movie_X_45_Y_180.mp4":  # Adjust the video file name accordingly
            video_path = os.path.join(video_folder, video_filename)
            extract_frames(video_path, output_folder)
