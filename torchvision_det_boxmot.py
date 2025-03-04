import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path

from boxmot import BotSort

# Load a pre-trained Faster R-CNN model
device = torch.device('cpu')  # Use 'cuda' if you have a GPU
detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
detector.eval().to(device)

# Initialize the tracker
tracker = BotSort(
    reid_weights=Path('osnet_x0_25_msmt17.pt'),  # Path to ReID model
    device=device,  # Use CPU for inference
    half=False
)

# Open the video file
vid = cv2.VideoCapture(0)  # or 'path/to/your.avi

while True:
    # Capture frame-by-frame
    ret, frame = vid.read()

    # If ret is False, it means we have reached the end of the video
    if not ret:
        break

    # Convert frame to tensor and move to device
    frame_tensor = torchvision.transforms.functional.to_tensor(frame).to(device)

    # Perform detection
    with torch.no_grad():
        detections = detector([frame_tensor])[0]

    # Filter the detections (e.g., based on confidence threshold)
    confidence_threshold = 0.5
    dets = []
    for i, score in enumerate(detections['scores']):
        if score >= confidence_threshold:
            bbox = detections['boxes'][i].cpu().numpy()
            label = detections['labels'][i].item()
            conf = score.item()
            dets.append([*bbox, conf, label])

    # Convert detections to numpy array (N X (x, y, x, y, conf, cls))
    dets = np.array(dets)

    # Update the tracker
    res = tracker.update(dets, frame)  # --> M X (x, y, x, y, id, conf, cls, ind)

    # Plot tracking results on the image
    tracker.plot_results(frame, show_trajectories=True)

    cv2.imshow('BoXMOT + Torchvision', frame)

    # Simulate wait for key press to continue, press 'q' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
vid.release()
cv2.destroyAllWindows()