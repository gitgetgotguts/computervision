# Hand Gesture Control System

A computer vision-based system that recognizes hand gestures and performs customized actions on your PC using MediaPipe for hand tracking and machine learning for gesture recognition.

## Overview

The system consists of four main components:

1. **`data.py`**: Captures hand gesture data using your webcam
2. **`data.ipynb`**: Processes, normalizes, and merges gesture data for training
3. **`train.py`**: Trains the machine learning model on the prepared data
4. **`main.py`**: Uses the trained model to perform actions based on recognized gestures

## Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Getting Started

1. Run the main application:
```bash
python main.py
```

2. Try the currently supported gestures:
   - **UP**: Increase system volume
   - **DOWN**: Decrease system volume
   - **Pinch gesture** (thumb & index finger close): Toggle mute

## Adding New Gestures

For detailed instructions on adding new gestures, see [GESTURES_GUIDE.md](GESTURES_GUIDE.md)

The helper script guides you through the entire process:

```bash
python add_gesture.py
```

### Quick Summary of Adding Gestures:

1. Record samples with `add_gesture.py` or `data.py`
2. Process data with `data.ipynb` (important: update labels correctly)
3. Train model with `train.py`
4. Update `main.py` with your new gesture action

### Working with Labels in data.ipynb

When adding gestures in `data.ipynb`, pay close attention to the labeling section:

1. You should only create specific labels for gestures you want to differentiate
2. Group all other gestures under the "OTHER" category (e.g., random hand positions)
3. Find the cell with `gesture_to_label` mapping and modify it like this:

```python
# Example: If you only want UP, DOWN and PEACE gestures to be distinct
gesture_to_label = {
    'up': 'UP',          # Specific gesture
    'down': 'DOWN',      # Specific gesture
    'peace': 'PEACE',    # Specific gesture
    'rand': 'OTHER',     # Grouped as OTHER
    'randi': 'OTHER',    # Grouped as OTHER
    'ping': 'OTHER',     # Grouped as OTHER
    # Add your new gesture here, deciding if it should be distinct or grouped as 'OTHER'
}
```

This approach improves recognition accuracy by teaching the model which gestures to distinguish and which to ignore.

## Why Normalization Matters

Hand landmark normalization (implemented in `data.ipynb`) is crucial for:

- **Position Independence**: Gesture recognition works regardless of where in the frame your hand appears
- **Scale Independence**: Works with different hand sizes and distances from camera
- **Better Generalization**: Model focuses on hand shape rather than absolute positions

## Tips for Creating Effective Gestures

1. Record multiple variations of each gesture (5-10 samples minimum)
2. Always include "random" or "other" gestures as negative examples
3. Make sure your gestures are visually distinct from each other
4. Record in different positions and angles for better recognition
5. Ensure good lighting for recording training data

## Cross-Platform Support

The system works on both Windows and Linux, automatically detecting your operating system and using the appropriate commands for audio control.

## Files and Configuration

- **MODEL_NAME**: "hand_gesture_model.pkl" (default model file)
- **TRAINING_DATA_PATH**: "training/unnormalized_data.csv" (training data location)

You can switch between different trained models by changing the MODEL_NAME variable in main.py.