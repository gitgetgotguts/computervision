# pip install mediapipe opencv-python
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.9)
mp_draw = mp.solutions.drawing_utils

# Helper: flatten landmarks into 63-value vector
def extract_landmark_vector(landmarks):
    return [coord
            for lm in landmarks.landmark
            for coord in (lm.x, lm.y, lm.z)]
def get_filename_via_cv():
    typed = ""
    while True:
        frame = 255 * np.ones((200, 600, 3), dtype=np.uint8)
        cv2.putText(frame, "Enter filename (ENTER to save, ESC to cancel):",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, typed,
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 255), 2)
        cv2.imshow("Filename Input", frame)
        key = cv2.waitKey(0)

        if key in [13, 10]:  # Enter
            cv2.destroyWindow("Filename Input")
            return typed.strip() + ".csv"
        elif key == 27:  # ESC
            cv2.destroyWindow("Filename Input")
            return None
        elif key == 8:  # Backspace
            typed = typed[:-1]
        elif 32 <= key <= 126:
            typed += chr(key)

# State
recording = False
data = []  # list of landmark vectors

print("Press 'i' to START/STOP recording; 'q' to quit.")
# Start webcam
webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    ret, frame = webcam.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            if recording:
                mp_draw.draw_landmarks(
        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),  # red landmarks
        mp_draw.DrawingSpec(color=(0, 0, 150), thickness=2)) 
                vec = extract_landmark_vector(hand_landmarks)
                data.append(vec)
            else:
                mp_draw.draw_landmarks(
        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # green landmarks
        mp_draw.DrawingSpec(color=(0, 150, 0), thickness=2))   



    # Show the frame
    cv2.imshow('Hand Tracking', frame)
    key = cv2.waitKey(2) & 0xFF

    if key == ord('i'):
        recording = not recording
        if recording:
            print("🔴 Recording started...")
        else:
            print("⏹️ Recording stopped.")
            if data:
                fname = get_filename_via_cv()
                if fname:
                    cols = [f'{axis}{i//3+1}' for i, axis in enumerate(['x','y','z']*21)]
                    df = pd.DataFrame(data, columns=cols)
                    df.to_csv(fname, index=False)
                    print(f"✅ Saved {len(data)} samples to {fname}")
                else:
                    print("❌ Save canceled.")
                data.clear()

    elif key == ord('q'):
        break

# Break the loop on 'q' key press
# if cv2.waitKey(2) & 0xFF == ord('q'):
#     break

webcam.release()
cv2.destroyAllWindows()