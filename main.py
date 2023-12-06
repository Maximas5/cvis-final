import cv2
import numpy as np
from Fire_Id import Fire_Id

def extract_frames(video_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frames = []
    while video_capture.isOpened():
        # Read a frame from the video file
        ret, frame = video_capture.read()

        # If there are no more frames to read, break the loop
        if not ret:
            break

        # Convert the frame to a NumPy array and append to the frames list
        frames.append(np.array(frame))

    # Release the video capture object
    video_capture.release()

    return frames

def main():
    # Import Data
    data = extract_frames('test_fire/indoor1.mp4')
    train = data
    test = data

    # Separate into train and test data (different cameras should not be used in the same train and/or test)


    # Make Fire_Id object
    classifier = Fire_Id()

    # Train
    classifier.train(train)

    # Test
    results = classifier.test(data)

    # Print results


if __name__ == "__main__":
    main()