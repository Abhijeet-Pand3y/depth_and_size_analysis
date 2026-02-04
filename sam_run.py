import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt


class SegmentAnythingModelRun:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)

        self.predictor = SamPredictor(sam)

    @torch.no_grad()
    def process_image(self, raw_img, x, y):
        # raw_image = cv2.imread(raw_img)
        self.predictor.set_image(raw_img)

        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False, # We only want the best mask for the ball
        )

        mask = masks[0]

        return mask
    
