import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path
from boxmot import BotSort

# Load a pre-trained Mask R-CNN model from torchvision
device = torch.device('cpu')  # Change to 'cuda' if you have a GPU available
segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
segmentation_model.eval().to(device)

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

    # Run the Mask R-CNN model to detect bounding boxes and masks
    with torch.no_grad():
        results = segmentation_model(frame_tensor)[0]

    # Extract detections (bounding boxes, masks, and scores)
    dets = []
    masks = []
    confidence_threshold = 0.5

    for i, score in enumerate(results['scores']):
        if score >= confidence_threshold:
            # Extract bounding box and score
            x1, y1, x2, y2 = results['boxes'][i].cpu().numpy()
            conf = score.item()
            cls = results['labels'][i].item()  # Assuming 'labels' represents the class
            dets.append([x1, y1, x2, y2, conf, cls])

            # Extract mask and add to list
            mask = results['masks'][i, 0].cpu().numpy()  # Use the first channel (binary mask)
            masks.append(mask)

    # Convert detections to a numpy array (N x (x, y, x, y, conf, cls))
    dets = np.array(dets)

    # Update tracker with detections and image
    tracks = tracker.update(dets, im)  # M x (x, y, x, y, id, conf, cls, ind)

    # Draw segmentation masks and bounding boxes in a single loop
    if len(tracks) > 0:
        inds = tracks[:, 7].astype('int')  # Get track indices as int

        # Use the indices to match tracks with masks
        if len(masks) > 0:
            masks = [masks[i] for i in inds if i < len(masks)]  # Reorder masks to match the tracks

        # Iterate over tracks and corresponding masks to draw them together
        for track, mask in zip(tracks, masks):
            track_id = int(track[4])  # Extract track ID
            color = get_color(track_id)  # Use unique color for each track

            # Draw the segmentation mask on the image
            if mask is not None:
                # Binarize the mask
                mask = (mask > 0.5).astype(np.uint8)

                # Blend mask color with the image
                im[mask == 1] = im[mask == 1] * 0.5 + np.array(color) * 0.5

            # Draw the bounding box
            x1, y1, x2, y2 = track[:4].astype('int')
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

            # Add text with ID, confidence, and class
            conf = track[5]
            cls = track[6]
            cv2.putText(im, f'ID: {track_id}, Conf: {conf:.2f}, Class: {cls}',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image
    cv2.imshow('Segmentation Tracking', im)

    # Break on pressing q or space
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()