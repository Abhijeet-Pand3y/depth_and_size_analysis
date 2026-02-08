import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import glob
import os
import yaml
import re
import csv
import joblib

from sam_run import SegmentAnythingModelRun
from depthanything import DepthAnythingRunner


class CalculateArea:
    def __init__(self, args):

        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_runner = SegmentAnythingModelRun() 
        self.depth_runner = DepthAnythingRunner(self.args)

    
    def get_coordinate(self, raw_img):

        coords = []
        def click_event(event, x, y, flags, params):
            # Check for left mouse click
            
            real_shape, resized_shape = params
            if event == cv2.EVENT_LBUTTONDOWN:
                real_x = int(x * (real_shape[1]/resized_shape[1]))
                real_y = int(y * (real_shape[0]/resized_shape[0]))

                coords.clear()
                coords.extend([real_x, real_y])

                print(f"Pixel Coordinates: x={real_x}, y={real_y}")
                
                cv2.circle(resized_img, (x, y), 2, (0, 255, 0), -1)
                cv2.imshow("Select Ball Center", resized_img)


        scale_percent = 20 
        width = int(raw_img.shape[1] * scale_percent / 100)
        height = int(raw_img.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized_img = cv2.resize(raw_img, dim, interpolation = cv2.INTER_AREA)

        real_shape = (raw_img.shape[0], raw_img.shape[1])
        resized_shape = (resized_img.shape[0], resized_img.shape[1])
        
        while not coords:
            cv2.namedWindow("Select Ball Center", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Select Ball Center", click_event, param=(real_shape, resized_shape))

            print("Click on the center of the ball. Press any key to confirm selection.")
            cv2.imshow("Select Ball Center", resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if not coords:
                print("No point selected! You must click on the ball before closing the window.")

        return coords[0], coords[1]
    
    def get_mask(self, raw_img):

        x, y = self.get_coordinate(raw_img)
        mask = self.sam_runner.process_image(raw_img, x, y)

        return mask
    
    def get_depth(self, raw_img):
        depth = self.depth_runner.process_image(raw_img)

        return depth

    def get_spatial_features(self, ball_contour, image_width, image_height):
        c_x, c_y = image_width / 2, image_height / 2
        
        M = cv2.moments(ball_contour)
        if M["m00"] != 0:
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(ball_contour)
            u, v = x + w/2, y + h/2


        raw_dist = np.sqrt((u - c_x)**2 + (v - c_y)**2)
        
        max_dist = np.sqrt(c_x**2 + c_y**2)
        normalized_dist = raw_dist / max_dist
        
        return normalized_dist

    def calculate_depth_and_area(self, raw_img):
        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        mask = self.get_mask(img_rgb)
        depth = self.get_depth(img_rgb)

        mask_uint8 = mask.astype(np.uint8) * 255

        kernel = np.ones((7, 7), np.uint8) 
        eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)

        safe_ball_pixels = depth[eroded_mask > 0]

        final_depth_val = np.median(safe_ball_pixels)
        print(f"Final Depth Value: {final_depth_val}")

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_contour = max(contours, key=cv2.contourArea)

        number_of_pixel = np.sum(mask)

        return final_depth_val, ball_contour, number_of_pixel

        



        
    
    def get_diameter(self, real_depth, ball_contour, pixel_count, focal_length_in_px):
        if len(ball_contour) >= 5:
            ellipse = cv2.fitEllipse(ball_contour)
            (x, y), (d1, d2), angle = ellipse

            d_major = max(d1, d2)
            d_minor = min(d1, d2)
            
            aspect_ratio = d_minor / d_major
            
            # If difference is > 5% (adjust 0.05 as needed)
            if (1.0 - aspect_ratio) > 0.2:

                pixel_diameter = d_major 
            else:
                pixel_diameter = (d1 + d2) / 2

        else:
            _, _, w, h = cv2.boundingRect(ball_contour)
            pixel_diameter = (w + h) / 2    

        ellipse_diameter = (pixel_diameter * real_depth) / focal_length_in_px


        pixel_diameter = 2 * np.sqrt(pixel_count / np.pi)
        circle_diameter = (pixel_diameter * real_depth) / focal_length_in_px

        return ellipse_diameter, circle_diameter
    
    def process_area_and_depth(self, raw_img):
        final_depth_val, ball_countour, number_of_pixel = self.calculate_depth_and_area(raw_img)
        distance_to_center = self.get_spatial_features(ball_countour, raw_img.shape[1], raw_img.shape[0])

        with open('./config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # slope = config['model_params']['slope']
        # intercept = config['model_params']['intercept']
        focal_length_in_px = config['model_params']['focal_length_px']

        # real_depth = slope * final_depth_val + intercept

        loaded_pipe = joblib.load('./regression/size_estimation_pipeline.pkl')

        new_obs = np.array([[final_depth_val, distance_to_center, np.log10(number_of_pixel)]])
        
        real_depth = loaded_pipe.predict(new_obs)[0]


        print(f"Real Depth: {real_depth}")

        ellipse_diameter, circle_diameter = self.get_diameter(real_depth, ball_countour,number_of_pixel, focal_length_in_px)

        info = {"ellipse_diameter":ellipse_diameter, 
                "circle_diameter": circle_diameter, 
                "real_depth" : real_depth, 
                "relative_depth": final_depth_val, 
                "number_of_pixel": number_of_pixel,
                "distance_to_center": distance_to_center,
                }

        return info

    def extract_image(self):
        if os.path.isdir(self.args.img_path):
            filenames = glob.glob(os.path.join(self.args.img_path, '*'))
        else:
            filenames = [self.args.img_path]

        os.makedirs(self.args.outdir, exist_ok=True)

        img_dict = {}

        for i, filename in enumerate(filenames):
            img = cv2.imread(filename)
            if img is None:
                continue
            key = re.findall(r'(\d+\.?\d*m?)\.jpg', filename)[0]
            img_dict[key] = self.process_area_and_depth(img)
        
        return img_dict

    def save_csv(self, img_dict, filename="./results/new_ball3.csv"):

        fieldnames = [
            'Image_ID', 
            'Ellipse_Diameter_m', 
            'Circle_Diameter_m', 
            'Real_Depth_m', 
            'Relative_Depth_Raw', 
            'Pixel_Count'
        ]

        try:
            with open(filename, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for key, val in img_dict.items():
                    writer.writerow({
                        'Image_ID': key,
                        'Ellipse_Diameter_m': float(val['ellipse_diameter']),
                        'Circle_Diameter_m': float(val['circle_diameter']),
                        'Real_Depth_m': float(val['real_depth']),
                        'Relative_Depth_Raw': float(val['relative_depth']),
                        'Pixel_Count': int(val['number_of_pixel'])
                    })
            print(f"Data successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
        
            
    
if __name__ == '__main__':
    with open('./config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    
    input_files = config['input_files']

    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--input-size', type=int, default=518)

    parser.add_argument('--img-path', type=str, default=input_files)
    parser.add_argument('--outdir', type=str, default='./vis_depth')

    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')

    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    calculateArea = CalculateArea(args)

    # diameter = calculateArea.process_area_and_depth()

    img_dict = calculateArea.extract_image()

    with open('./config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    
    save_file = False
    if save_file:
        calculateArea.save_csv(img_dict)


    for key, val in img_dict.items():
        print(f"{key}: {val}")

    





