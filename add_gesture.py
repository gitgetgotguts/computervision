#!/usr/bin/env python3
"""
Hand Gesture Addition Helper Script
----------------------------------
This script automates the process of adding new gestures to the hand gesture recognition system.
It handles recording, data processing, model training, and updating necessary files.
"""

import os
import sys
import subprocess
import time
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Constants
MODEL_NAME = "hand_gesture_model.pkl"
TRAINING_DIR = "training"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    print("=" * 80)
    print("                       HAND GESTURE ADDITION HELPER                       ")
    print("=" * 80)
    print()

def get_existing_gestures():
    """Get the list of gestures from the current model"""
    try:
        model = joblib.load(MODEL_NAME)
        # Get existing label_map from main.py
        with open('main.py', 'r') as f:
            content = f.read()
            
        # Try to extract label_map
        import re
        match = re.search(r'label_map\s*=\s*{([^}]+)}', content)
        if match:
            label_map_str = match.group(1)
            # Convert string representation to dictionary
            label_map = {}
            for item in label_map_str.split(','):
                if ':' in item:
                    k, v = item.split(':')
                    try:
                        k = int(k.strip())
                        v = v.strip().strip('"\'')
                        label_map[k] = v
                    except:
                        pass
            return label_map
        else:
            print("Couldn't parse label_map from main.py, using default")
            return {0: "UP", 1: "DOWN", 2: "OTHER"}
    except:
        print("Couldn't load model, using default gestures")
        return {0: "UP", 1: "DOWN", 2: "OTHER"}

def record_gesture(gesture_name):
    """Record examples of the new gesture"""
    print_header()
    print(f"Recording samples for gesture: {gesture_name}")
    print("\nPosition your hand in front of the camera and make the gesture.")
    print("Press 'i' to START recording")
    print("Press 'i' again to STOP recording")
    print("Press 'q' to quit without saving")
    print("\nPRO TIP: Record at least 5-10 samples with slight variations for better results!")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.9)
    mp_draw = mp.solutions.drawing_utils

    def extract_landmark_vector(landmarks):
        return [coord for lm in landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

    recording = False
    data = []  # list of landmark vectors

    webcam = cv2.VideoCapture(0)
    countdown = 0

    while webcam.isOpened():
        ret, frame = webcam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Add clear instructions on the frame
        if not recording and countdown == 0:
            cv2.putText(frame, "Press 'i' to START recording", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif recording:
            cv2.putText(frame, "RECORDING... Press 'i' to STOP", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Samples: {len(data)}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if recording:
                    vec = extract_landmark_vector(hand_landmarks)
                    data.append(vec)
                    
        # Display countdown when applicable
        if countdown > 0:
            cv2.putText(frame, f"Starting in: {countdown}", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Hand Gesture Recording', frame)
        key = cv2.waitKey(2) & 0xFF

        # Handle countdown
        if countdown > 0:
            countdown -= 1
            if countdown == 0:
                recording = True
                print("🔴 Recording started...")

        if key == ord('i'):
            if not recording:
                countdown = 3  # 3 frame countdown
                print("Ready...")
            else:
                recording = False
                print("⏹️ Recording stopped.")
                
                # Save the data
                fname = os.path.join(TRAINING_DIR, f"{gesture_name.lower()}.csv")
                cols = [f'{axis}{i//3+1}' for i, axis in enumerate(['x','y','z']*21)]
                df = pd.DataFrame(data, columns=cols)
                df.to_csv(fname, index=False)
                print(f"✅ Saved {len(data)} samples to {fname}")
                data.clear()
                break

        elif key == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
    time.sleep(1)

def update_main_file(new_gesture_name, label_id, model_filename):
    """Update the label_map in main.py"""
    try:
        # Ask if the user wants to update main.py at all
        update_main = input("\nDo you want to update main.py with your new gesture? (y/n): ").lower() == 'y'
        
        if not update_main:
            print(f"ℹ️ No changes made to main.py.")
            print(f"To use your new gesture, manually add '{label_id}: \"{new_gesture_name}\"' to the label_map in main.py.")
            return True
            
        with open('main.py', 'r') as f:
            content = f.readlines()
        
        # Update MODEL_NAME if it's different from the default
        if model_filename != MODEL_NAME:
            update_model = input("Do you want to update main.py to use this new model? (y/n): ").lower() == 'y'
            if update_model:
                for i, line in enumerate(content):
                    if line.strip().startswith('MODEL_NAME = '):
                        content[i] = f'MODEL_NAME = "{model_filename}"\n'
                        break
                print(f"✅ Will update MODEL_NAME to: {model_filename}")
            else:
                print(f"ℹ️ Keeping current model in main.py. To use the new model, manually update MODEL_NAME.")
        
        # Update the label map
        for i, line in enumerate(content):
            if 'label_map = {' in line:
                # Find the end of the label_map dictionary
                j = i
                while j < len(content) and '}' not in content[j]:
                    j += 1
                    
                if j < len(content):
                    # Replace the closing brace with the new entry and closing brace
                    content[j] = content[j].replace('}', f', {label_id}: "{new_gesture_name}"}}')
                    break
        
        with open('main.py', 'w') as f:
            f.writelines(content)
        print(f"✅ Updated main.py with new gesture: {new_gesture_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to update main.py: {e}")
        print("You'll need to manually add the gesture to label_map in main.py")
        return False

def provide_jupyter_instructions(gesture_name):
    """Provide instructions for running the Jupyter notebook manually"""
    print("\n" + "=" * 80)
    print("           MANUAL DATA PROCESSING INSTRUCTIONS")
    print("=" * 80)
    print("\nTo process the recorded data:")
    print("1. Start Jupyter Notebook:")
    print("   jupyter notebook")
    print("\n2. Open training/data.ipynb")
    
    print("\n3. IMPORTANT: Modify the gesture labels appropriately")
    print("   Find the cell with the gesture_to_label mapping (usually a few cells down)")
    print("   Update it to include your new gesture:")
    print("   ```")
    print("   gesture_to_label = {")
    print("       'up': 'UP',")
    print("       'down': 'DOWN',")
    print(f"       '{gesture_name.lower()}': '{gesture_name}',  # Your new gesture")
    print("       'rand': 'OTHER',  # Group as OTHER")
    print("       'randi': 'OTHER', # Group as OTHER")
    print("       # Add other gestures here")
    print("   }")
    print("   ```")
    
    print("\n   LABELING STRATEGY:")
    print("   - Only create distinct labels for gestures you want to recognize as commands")
    print("   - Group all other gestures as 'OTHER' to help the model learn what to ignore")
    print("   - This improves accuracy by providing clear positive and negative examples")
    
    print("\n4. Run all cells in the notebook to process and normalize the data")
    print("\n5. Make sure to save the notebook after execution")
    print("\nAfter completing these steps, return to this script and continue.")

def train_model(custom_name=None):
    """Train the model with the new data"""
    try:
        print("Training model...")
        train_script = os.path.join(TRAINING_DIR, 'train.py')
        
        # If custom model name is provided, modify train.py temporarily
        temp_backup = None
        if custom_name:
            # Create a backup of train.py
            with open(train_script, 'r') as f:
                temp_backup = f.read()
            
            # Modify train.py to use custom model name
            with open(train_script, 'r') as f:
                content = f.read()
            
            # Replace model filename
            content = content.replace(f'"{MODEL_NAME}"', f'"{custom_name}"')
            content = content.replace(f"'{MODEL_NAME}'", f"'{custom_name}'")
            
            with open(train_script, 'w') as f:
                f.write(content)
        
        # Run the training
        result = subprocess.run(['python', train_script],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Restore the original train.py if needed
        if temp_backup:
            with open(train_script, 'w') as f:
                f.write(temp_backup)
        
        if result.returncode == 0:
            print("✅ Successfully trained model")
            model_path = custom_name if custom_name else MODEL_NAME
            print(f"✅ Model saved as: {model_path}")
            return True
        else:
            print("❌ Failed to train model. Error:")
            print(result.stderr.decode())
            return False
    except Exception as e:
        print(f"❌ Failed to train model: {e}")
        return False

def add_action_instructions(gesture_name):
    """Provide instructions on adding action code for the new gesture"""
    print("\n" + "=" * 80)
    print(f"ADDING ACTION FOR '{gesture_name}'")
    print("=" * 80)
    print("\nTo make your new gesture perform an action, edit main.py:")
    print("\n1. Find the section where gestures are handled (look for 'if text==\"UP\"')")
    print("2. Add this code:")
    print(f"""
    elif text=="{gesture_name}":
        print("{gesture_name} detected!")
        # Add your action code here
        # For example:
        # if is_linux:
        #     subprocess.run(["xdg-open", "https://example.com"])
        # else:
        #     import webbrowser
        #     webbrowser.open("https://example.com")
    """)

def main():
    print_header()
    print("This helper will guide you through adding a new hand gesture to the system.")
    print("It will record gesture samples and help you train the model.\n")

    # Get existing gestures
    existing_gestures = get_existing_gestures()
    print("Current gestures:")
    for idx, name in existing_gestures.items():
        print(f"  {idx}: {name}")
    print()
    
    # Get new gesture name
    gesture_name = input("Enter the name for your new gesture (e.g., SWIPE_LEFT): ").upper()
    if gesture_name in existing_gestures.values():
        overwrite = input(f"⚠️ Gesture '{gesture_name}' already exists! Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Aborted. No changes made.")
            return
    
    # Find the next available index
    next_idx = max(existing_gestures.keys()) + 1 if existing_gestures else 0
    
    # Record gesture
    input("\nPress Enter to start recording samples...")
    record_gesture(gesture_name)
    
    # Provide instructions for data processing instead of auto-running
    provide_jupyter_instructions(gesture_name)
    input("\nPress Enter when you've completed the data processing steps...")
    
    # Ask if user wants to create a new model version
    custom_model = input("\nDo you want to create a new model version instead of overwriting? (y/n): ").lower() == 'y'
    model_filename = MODEL_NAME
    
    if custom_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"hand_gesture_model_{timestamp}.pkl"
        print(f"New model will be saved as: {model_filename}")
    
    # Train the model
    input("\nPress Enter to train the model...")
    if custom_model:
        train_model(model_filename)
    else:
        train_model()
    
    # Update main.py
    update_main_file(gesture_name, next_idx, model_filename)
    
    # Provide instructions for adding the action code
    add_action_instructions(gesture_name)
    
    print("\n" + "=" * 80)
    print("✨ All done! Your new gesture has been added to the system.")
    print("=" * 80)

if __name__ == "__main__":
    main()