import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path
from boxmot import BotSort

# Load a pre-trained Keypoint R-CNN model from torchvision
device = torch.device('cpu')  # Change to 'cuda' if you have a GPU available
pose_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
pose_model.eval().to(device)

tracker = BotSort(
    reid_weights=Path('osnet_x0_25_msmt17.pt'),  # ReID model to use
    device=device,
    half=False,
)

# Open the video file
vid = cv2.VideoCapture(0)

# Function to generate a unique color for each track ID
def get_color(track_id):
    np.random.seed(int(track_id))
    return tuple(np.random.randint(0, 255, 3).tolist())

while True:
    ret, im = vid.read()
    if not ret:
        break

    # Convert frame to tensor and move to device
    frame_tensor = torchvision.transforms.functional.to_tensor(im).unsqueeze(0).to(device)

    # Run the Keypoint R-CNN model to detect keypoints and bounding boxes
    with torch.no_grad():
        results = pose_model(frame_tensor)[0]

    # Extract detections (bounding boxes and keypoints)
    dets = []
    keypoints = []

    confidence_threshold = 0.5
    for i, score in enumerate(results['scores']):
        if score >= confidence_threshold:
            # Extract bounding box and score
            x1, y1, x2, y2 = results['boxes'][i].cpu().numpy()
            conf = score.item()
            cls = results['labels'][i].item()  # Assuming that 'labels' would be person class
            dets.append([x1, y1, x2, y2, conf, cls])

            # Extract keypoints
            keypoint = results['keypoints'][i].cpu().numpy().tolist()
            keypoints.append(keypoint)

    # Convert detections to a numpy array (N x (x, y, x, y, conf, cls))
    dets = np.array(dets)

    # Update tracker with detections and image
    tracks = tracker.update(dets, im)  # M x (x, y, x, y, id, conf, cls, ind)

    if len(tracks) > 0:
        inds = tracks[:, 7].astype('int')  # Get track indices as int

        # Use the indices to match tracks with keypoints
        keypoints = [keypoints[i] for i in inds if i < len(keypoints)]  # Reorder keypoints to match the tracks

        # Draw bounding boxes and keypoints in the same loop
        for i, track in enumerate(tracks):
            x1, y1, x2, y2, track_id, conf, cls = track[:7].astype('int')
            color = get_color(track_id)

            # Draw bounding box with unique color
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

            # Add text with ID, confidence, and class
            cv2.putText(im, f'ID: {track_id}, Conf: {conf:.2f}, Class: {cls}',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw keypoints for the corresponding track
            if i < len(keypoints):
                kp = keypoints[i]
                for point in kp:
                    x, y, confidence = int(point[0]), int(point[1]), point[2]
                    if confidence > 0.5:  # Only draw keypoints with confidence > 0.5
                        cv2.circle(im, (x, y), 3, color, -1)  # Draw keypoints in the color of the corresponding track

    # Display the image
    cv2.imshow('Pose Tracking', im)

    # Break on pressing q or space
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()