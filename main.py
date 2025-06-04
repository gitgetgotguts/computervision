# pip install mediapipe opencv-python
import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyautogui
import subprocess
import platform
import time

MODEL_NAME = "mhand_gesture_model.pkl"
# === 1) Load your trained model (once) ===
model = joblib.load(MODEL_NAME)   # ← ADDED
label_map = {0: "UP", 1: "DOWN", 2: "OTHER"}    # add the labels maps just like in the data.ipynb


last_action_time = 0
cooldown_period = 1.0  # 1 second cooldown between actions

is_linux = platform.system() == "Linux"
# For Windows audio control
if not is_linux:
    # pip install pycaw
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    import math
def get_volume_info():
    """Get current volume level and mute status"""
    if is_linux:
        try:
            # Get volume information using amixer
            result = subprocess.check_output(["amixer", "-D", "pulse", "get", "Master"]).decode()
            
            # Extract volume percentage
            volume = "0%"
            for line in result.splitlines():
                if "%" in line:
                    volume = line.split("[")[1].split("]")[0]
                    break
            
            # Check if muted
            is_muted = "[off]" in result
            
            return f"Volume: {volume}", is_muted
        except Exception as e:
            print(f"Error getting volume info: {e}")
            return "Volume: Error", False
    else:
        # Windows implementation using pycaw
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            
            # Get volume level (0.0 to 1.0)
            level = volume.GetMasterVolumeLevelScalar()
            vol_percent = f"{int(level * 100)}%"
            
            # Get mute state
            muted = volume.GetMute()
            
            return f"Volume: {vol_percent}", muted
        except Exception as e:
            print(f"Error getting Windows volume: {e}")
            return "Volume: Error", False
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.9)
mp_draw = mp.solutions.drawing_utils



# === 2) Normalization function (must match training) ===
def normalize_landmarks(flat_landmarks):        # ← ADDED
    lm = np.array(flat_landmarks, dtype=float).reshape(21, 3)
    wrist = lm[0].copy()
    lm -= wrist
    scale = np.linalg.norm(lm[12])
    if scale > 0:
        lm /= scale
    return lm.flatten()


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
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        # --- EXTRACT, NORMALIZE & PREDICT ---
            # Flatten to [x1,y1,z1,...,x21,y21,z21]
            flat = [coord
                    for lm in hand_landmarks.landmark
                    for coord in (lm.x, lm.y, lm.z)]
            norm = normalize_landmarks(flat)              # normalize     ← ADDED
            pred = model.predict([norm])[0]               # classify      ← ADDED
            text = label_map[pred]  
                   # map to text   ← ADDED
            # Overlay classification result
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # ← ADDED

            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate 2D distance (normalized)
            distance = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5

            # Trigger drawing if fingers are close
            if distance < 0.08:  # Adjust threshold as needed TRAILLLLLLLLLLLLLLLLLLLLLL
                # Draw a circle at the tip of the index finger
                for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks with a thick outline
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=5),  # Outline
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)   # Inner lines
                    )
                current_time = time.time()
    # Only trigger if enough time has passed since the last action
                if current_time - last_action_time > cooldown_period:
                    # Draw a circle at the tip of the index finger
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Draw hand landmarks with a thick outline
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=5),  # Outline
                            mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)   # Inner lines
                        )
                    if is_linux:
                        subprocess.run(["amixer", "-D", "pulse", "set", "Master", "toggle"])
                    else:
                        pyautogui.press("volumemute")
                    
                    # Update the last action time
                    last_action_time = current_time
                    # Show visual feedback
                    cv2.putText(frame, "MUTE TOGGLED", (200, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


            else:
                if text=="UP":
                                        # Night vision effect for the whole frame
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Apply a green color map
                    night_vision = cv2.applyColorMap(gray, cv2.COLORMAP_SUMMER)
                    # Optionally boost brightness
                    night_vision = cv2.convertScaleAbs(night_vision, alpha=1.2, beta=30)
                    # Overlay text for feedback
                    cv2.putText(night_vision, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    frame[:] = night_vision  # Replace the frame with the effect
                    # print("Volume Up")
                    # if is_linux:
                    #     subprocess.run(["amixer", "-D", "pulse", "set", "Master", "5%+"])
                    # else:
                    #     pyautogui.press("volumeup")
                elif text=="DOWN":
                    if is_linux:
                        subprocess.run(["amixer", "-D", "pulse", "set", "Master", "5%-"])
                    else:
                        pyautogui.press("volumedown")
                # Get and display volume information
            volume_text, is_muted = get_volume_info()
            cv2.putText(frame, volume_text, (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display mute status
            mute_status = "MUTED" if is_muted else "UNMUTED"
            mute_color = (0, 0, 255) if is_muted else (0, 255, 0)  # Red if muted, green if unmuted
            cv2.putText(frame, mute_status, (200, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, mute_color, 2)    

    # Show the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()