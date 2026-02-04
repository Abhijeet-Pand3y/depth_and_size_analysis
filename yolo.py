from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import os

class CalculateArea:
    def __init__(self, model_path='./checkpoints/yolo12s.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize SAHI with the updated from_pretrained call
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8', 
            model_path=model_path,
            confidence_threshold=0.2,
            device=self.device
        )
        print(f"SAHI + YOLO12 initialized on: {self.device}")

    def get_ball_box_sahi(self, image_path):
        """Finds the ball and returns [x1, y1, x2, y2]."""
        result = get_sliced_prediction(
            image_path,
            self.detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",
            verbose=0
        )

        for object_prediction in result.object_prediction_list:
            if object_prediction.category.id == 32: # Sports Ball
                return object_prediction.bbox.to_xyxy()
        return None

    def save_visual(self, image_path, box, output_name="sahi_test_result.jpg"):
        """Draws the box and saves it so you can verify the 'lock'."""
        img = cv2.imread(image_path)
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            # Draw a bright green box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(img, "Ball Detected", (x1, y1 - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imwrite(output_name, img)
            print(f"Visual saved to: {output_name}")
        else:
            print("No box to draw. Detection failed.")

if __name__ == '__main__':
    calc = CalculateArea('./checkpoints/yolo12s.pt')
    
    test_image_path = './input_images/0.6m.jpg' 
    
    print(f"Running SAHI inference on {test_image_path}...")
    ball_box = calc.get_ball_box_sahi(test_image_path)
    
    if ball_box is not None:
        print(f"Success! Box Coordinates: {ball_box}")
        # Save the image to see the result
        calc.save_visual(test_image_path, ball_box)
    else:
        print("Detection failed. Try lowering confidence_threshold in __init__.")