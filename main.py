# main.py — debug + stable detector
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
import time
import os
import re
try:
    from playsound import playsound
except Exception:
    playsound = None

# Path settings (absolute to avoid path mistakes)
BASE_DIR = r"D:\Animal_Detector"
MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
ALERT_PATH = os.path.join(BASE_DIR, "alert.mp3")

print("Base dir:", BASE_DIR)
print("Model:", MODEL_PATH)
print("Labels:", LABELS_PATH)
print("Alert:", ALERT_PATH)

# ----- sanity check: alert file exists -----
if not os.path.isfile(ALERT_PATH):
    print("ERROR: alert.mp3 not found at", ALERT_PATH)
    print("-> Place alert.mp3 in the folder and re-run.")
    # we will still proceed, but no sound will play
else:
    print("alert.mp3 found ✔")

# ----- load and clean labels -----
raw_labels = []
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    raw_labels = [line.strip() for line in f.readlines() if line.strip()]

def clean_label(s):
    # Remove leading numbers and non-letter characters, keep rest
    s = s.strip()
    # If label like "0 Elephant" -> remove leading "0 "
    s = re.sub(r'^[\d\W_]+', '', s)   # remove leading digits/punctuation/underscores
    return s

class_names = [clean_label(l) for l in raw_labels]
print("Loaded labels (cleaned):", class_names)

# Define targets (must match cleaned labels)
TARGET_CLASSES = ["Elephant", "Lion", "Cheetah", "Tiger", "Bear", "Gorilla", "Rhinosaur", "Hyena"]

# Verify target classes exist in labels
missing_targets = [t for t in TARGET_CLASSES if t not in class_names]
if missing_targets:
    print("WARNING: These TARGET_CLASSES are not present in labels.txt:", missing_targets)
    print("Make sure labels order/names match your Teachable Machine export.")
else:
    print("All target classes found in labels ✔")

# Load model
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded.")

# Playsound wrapper (non-blocking)
def play_alert():
    if playsound is None:
        print("playsound not available — can't play audio.")
        return
    # Play in background thread so main loop isn't blocked
    threading.Thread(target=playsound, args=(ALERT_PATH,), daemon=True).start()

# Basic live test function (single frame prediction)
def single_test():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera for single_test.")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Failed to read a frame for single_test.")
        return
    img = cv2.resize(frame, (224,224))
    x = (np.asarray(img, dtype=np.float32).reshape(1,224,224,3) / 127.5) - 1
    preds = model.predict(x, verbose=0)[0]
    # Print top 3 predictions
    top_idx = np.argsort(preds)[-3:][::-1]
    print("Single test top-3 predictions:")
    for i in top_idx:
        print(f"  {class_names[i]} -> {preds[i]:.4f}")
    return

print("Running a quick single-frame test (check if model sees something)...")
single_test()

# ---- main loop ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam. Exiting.")
    exit(1)

print("Starting live detection. Press 'q' to quit.")
last_alert = 0
cooldown = 2.5   # seconds between alert sounds
history = []
HISTORY_LEN = 5
CONF_THRESHOLD = 0.65  # start lower to not miss; we'll print confidences

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed. Breaking.")
        break

    img = cv2.resize(frame, (224,224))
    x = (np.asarray(img, dtype=np.float32).reshape(1,224,224,3) / 127.5) - 1

    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    raw_label = raw_labels[idx] if idx < len(raw_labels) else str(idx)
    label = class_names[idx] if idx < len(class_names) else raw_label

    # Keep history of raw predicted labels (cleaned)
    history.append(label)
    if len(history) > HISTORY_LEN:
        history.pop(0)
    # majority vote
    smoothed = max(set(history), key=history.count)

    # Draw text
    text = f"{smoothed} {conf*100:.1f}%"
    cv2.putText(frame, text, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Detector DEBUG", frame)

    # DEBUG PRINT every frame (or throttle)
    # Print only if confidence high or label is a target, so we don't spam console
    if conf > 0.30 or label in TARGET_CLASSES:
        # Print top-3 for context when needed
        top_idx = np.argsort(preds)[-3:][::-1]
        top3 = ", ".join([f"{class_names[i]}:{preds[i]:.3f}" for i in top_idx])
        print(f"[{time.strftime('%H:%M:%S')}] Predicted: {label} ({conf:.3f}) | top3: {top3}")

    # SOUND logic: only sound if the smoothed label is a target AND confidence above threshold AND cooldown passed
    if (smoothed in TARGET_CLASSES) and (conf >= CONF_THRESHOLD) and (time.time() - last_alert > cooldown):
        print(f"*** ALERT SOUND (target={smoothed}, conf={conf:.3f}) ***")
        play_alert()
        last_alert = time.time()

    # exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Exited cleanly.")