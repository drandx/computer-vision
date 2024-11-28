import cv2
import mediapipe as mp
import time

# Instructions to run the script:
# 1. Ensure you have OpenCV and MediaPipe installed: `pip install opencv-python mediapipe`
# 2. Run the script using Python: `python actions_detection.py`
# 3. The script will open a webcam video stream and display detected actions (hands up, hands down, jump) on the screen.
# 4. Detected actions will also be written to a file named `actions.txt`.

# Initialize MediaPipe pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize the webcam video stream
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect key points
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get the coordinates of the shoulders and hands
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Check for hands up action
        if left_hand.y < left_shoulder.y and right_hand.y < right_shoulder.y:
            action = "Hands Up"
        # Check for hands down action
        elif left_hand.y > left_shoulder.y and right_hand.y > right_shoulder.y:
            action = "Hands Down"
        # Check for jump action
        elif left_shoulder.y < 0.5 and right_shoulder.y < 0.5:
            action = "Jump"
        else:
            action = "None"

        # Display the detected action on the screen
        cv2.putText(frame, action, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the detected action to a file
        with open("actions.txt", "a") as f:
            f.write(f"{action}\n")

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
