import cv2
import numpy as np
from Fire_Id import Fire_Id
import pandas as pd

def extract_frames(video_path, limit=9999):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frames = []
    count = 0
    while video_capture.isOpened() and count < limit:
        # Read a frame from the video file
        ret, frame = video_capture.read()

        # If there are no more frames to read, break the loop
        if not ret:
            break

        # Convert the frame to a NumPy array and append to the frames list
        frames.append(np.array(frame))

        count += 1

    # Release the video capture object
    video_capture.release()

    return frames

def main(verbose=True):
    # Import Data
    print("Extracting Frames")
    video = extract_frames('test_fire/indoor640x360-30fps.mp4', 5400)

    # Make Fire_Id object
    classifier = Fire_Id()

    # Test
    testStart = 30

    # Train
    print(f'Training')
    train = video[:testStart]
    classifier.train(train)

    # Get Results
    results = pd.DataFrame(columns=["Frame", "Cce", "Cse", "Cmes"])
    # For each frame in test...
    for t in range(testStart, len(video)):
        print(f'Frame {t}')
        # Predict
        Cce, Cse, Cmes = classifier.predict(video[t], video[t-1], verbose=verbose)
        # Log prediction and frame
        results.loc[len(results.index)] = [t, Cce, Cse, Cmes]

    # Display results
    results.to_csv('data/fire_class.csv')

if __name__ == "__main__":
    main(False)