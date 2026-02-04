import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Setup Device & Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# --- 2. Load Image ---
filename = '11.jpg'
image_path = f'./input_images/test/demo/{filename}'
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not find image at {image_path}")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# --- 3. Interactive Point Selection ---
coords = []

def click_event(event, x, y, flags, params):
    real_shape, resized_shape = params
    if event == cv2.EVENT_LBUTTONDOWN:
        # Map resized coordinates back to original image scale
        real_x = int(x * (real_shape[1] / resized_shape[1]))
        real_y = int(y * (real_shape[0] / resized_shape[0]))
        
        coords.append((real_x, real_y))
        print(f"Added Point: Original x={real_x}, y={real_y}")
        
        # Visual feedback on the window
        cv2.circle(resized_img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points (Press any key to finish)", resized_img)

# Scaling logic for display
scale_percent = 20 
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

resized_img = image.copy()
resized_img = cv2.resize(resized_img, dim, interpolation=cv2.INTER_AREA)

real_shape = (image.shape[0], image.shape[1])
resized_shape = (resized_img.shape[0], resized_img.shape[1])

cv2.namedWindow("Select Points (Press any key to finish)", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select Points (Press any key to finish)", click_event, param=(real_shape, resized_shape))

print("Click on the centers. Press any key to start SAM segmentation.")
cv2.imshow("Select Points (Press any key to finish)", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- 4. SAM Prediction ---
os.makedirs('./sam_output/', exist_ok=True)

if not coords:
    print("No points selected. Exiting.")
else:
    print(f"Processing {len(coords)} points...")
    for x, y in coords:
        input_point = np.array([[x, y]])
        input_label = np.array([1]) # 1 indicates a foreground point

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask = masks[0]
        
        # Visualization: Create a green overlay
        segmented_vis = image.copy()
        # We use a weighted overlay so you can still see the object underneath
        overlay = segmented_vis.copy()
        overlay[mask] = [0, 255, 0] 
        cv2.addWeighted(overlay, 0.5, segmented_vis, 0.5, 0, segmented_vis)

        output_filename = f'mask_{x}_{y}.png'
        # Saving using BGR -> RGB correction for plt.imsave or just use cv2.imwrite
        cv2.imwrite(f'./sam_output/{output_filename}', segmented_vis)
        
        num_pixels = np.sum(mask)
        print(f"Saved: {output_filename} | Object Size: {num_pixels} pixels")

print("Processing complete.")