# Hand Gesture Recognition System - Gesture Guide

This guide provides detailed instructions for adding new gestures to the hand gesture recognition system.

## Overview of Gesture Addition Process

Adding a new gesture involves four main steps:

1. **Recording** samples of the new gesture
2. **Processing** the data, including normalization and labeling
3. **Training** a new or updated model with the gesture
4. **Using** your new gesture in the application

## Step 1: Recording Gesture Data

### Option A: Using the Helper Script (Recommended)

```bash
python add_gesture.py
```

The helper will guide you through the entire process, including recording.

### Option B: Using data.py Directly

```bash
python data.py
```

- Press 'i' to START recording
- Press 'i' again to STOP recording
- Enter a gesture name when prompted
- Data is saved to `training/[gesture_name].csv`

## Step 2: Processing Data with data.ipynb

This is the **critical step** where you define how your gestures are recognized.

1. Open the notebook:
   ```bash
   cd training
   jupyter notebook data.ipynb
   ```

2. **Important: Managing Gesture Labels**

   Locate the cell that defines the `gesture_to_label` mapping (usually a few cells down in the notebook).
   
   ```python
   # Original mapping may look like this
   gesture_to_label = {
       'up': 'UP',
       'down': 'DOWN',
       'rand': 'OTHER',
       'randi': 'OTHER'
   }
   ```

3. **Update the labels appropriately:**

   ```python
   # Updated mapping with your new gesture
   gesture_to_label = {
       'up': 'UP',          # Specific important gesture
       'down': 'DOWN',      # Specific important gesture
       'peace': 'PEACE',    # Your new specific gesture
       'rand': 'OTHER',     # Group as OTHER
       'randi': 'OTHER',    # Group as OTHER
       'new': 'OTHER',      # Group as OTHER
   }
   ```

### Label Management Strategy

* **Only create distinct labels** (like 'UP', 'DOWN', 'PEACE') for gestures you want to actually use in your application
* **Group all other gestures** as 'OTHER' - this helps the model learn what is NOT a command gesture
* **Be consistent** with your labeling across training sessions

### Why This Approach Works Better

1. **Improved Accuracy**: The model learns to distinguish between meaningful gestures and random hand positions
2. **Reduced Confusion**: Fewer categories means less potential overlap between gesture classifications
3. **Better Negative Examples**: The 'OTHER' category teaches the model to ignore irrelevant hand positions

## Step 3: Training the Model

After updating the labels in the notebook, run all cells to process the data.

Then train the model:
```bash
cd training
python train.py
```

You can create a new model version by modifying the output filename in `train.py` or use the helper script's versioning feature.

## Step 4: Updating main.py

After training, you need to:

1. Update the `label_map` dictionary in `main.py` to match your gestures:

   ```python
   # If you added a PEACE gesture with ID 3
   label_map = {0: "UP", 1: "DOWN", 2: "OTHER", 3: "PEACE"}
   ```

2. Add code to handle your new gesture:

   ```python
   elif text=="PEACE":
       print("Peace sign detected!")
       # Add your action code here
       if is_linux:
           subprocess.run(["xdg-open", "https://www.example.com"])
       else:
           import webbrowser
           webbrowser.open("https://www.example.com")
   ```

## Tips for Creating Good Gesture Samples

1. **Record 5-10 samples** minimum per gesture
2. **Vary slightly** in position, angle, and hand shape
3. **Record from different distances** from the camera
4. **Include "random" gestures** as negative examples
5. **Use good lighting**
6. **Keep gestures visually distinct** from each other
7. **Hold each gesture steady** during recording

## Example Workflow: Adding a "Peace Sign" Gesture

1. Run `add_gesture.py` and name your gesture "PEACE"
2. Record 5-10 variations of your peace sign
3. Open and run `data.ipynb`, ensuring the mapping includes:
   ```python
   gesture_to_label = {
       'up': 'UP',
       'down': 'DOWN',
       'peace': 'PEACE',  # New gesture as distinct
       'rand': 'OTHER',
       'randi': 'OTHER'
   }
   ```
4. Train the model with `train.py`
5. Update `main.py` with your new label and action code

## Troubleshooting

* **Recognition Problems**: Record more samples with greater variation
* **Confused Gestures**: Make sure your gestures are visually distinct
* **Model Performance**: Try adding more "OTHER" samples for better differentiation
* **Mislabeled Data**: Double-check your gesture_to_label mapping in data.ipynb