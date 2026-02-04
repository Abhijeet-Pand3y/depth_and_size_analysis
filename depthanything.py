import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

class DepthAnythingRunner:
    def __init__(self, args):
        
        self.args = args


        # Devide setup
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
        

        #Load Model
        self.depth_model = DepthAnythingV2(**model_configs[args.encoder])
        self.depth_model.load_state_dict(torch.load(self.args.load_from, map_location='cpu'))
        self.depth_model = self.depth_model.to(self.device).eval()

        #Choosing color maps for output image
        self.cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    def process_image(self, raw_img):
        # raw_image = cv2.imread(raw_img)

        with torch.no_grad():
                depth = self.depth_model.infer_image(raw_img, input_size=self.args.input_size)

        return depth


    def process_and_save_image(self):
        if os.path.isdir(self.args.img_path):
            filenames = glob.glob(os.path.join(self.args.img_path, '*'))
        else:
            filenames = [self.args.img_path]

        os.makedirs(self.args.outdir, exist_ok=True)

        for i, filename in enumerate(filenames):
        
            raw_image = cv2.imread(filename)
            if raw_image is None:
                continue
            
            with torch.no_grad():
                depth = self.depth_model.infer_image(raw_image, input_size=self.args.input_size)

            # Save npy file befor normalization
            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_path = os.path.join(self.args.outdir, f"{base_name}.npy")
            np.save(output_path, depth)

            output_path = os.path.join(self.args.outdir, f'{base_name}_vis.png')

            # min max normalization
            depth_range = depth.max() - depth.min()

            #Normalize safely (adding 1e-6 prevents division by zero)
            depth = (depth - depth.min()) / (depth_range + 1e-6) * 255.0
            depth = depth.astype(np.uint8)

            if self.args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (self.cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            if args.pred_only:
                cv2.imwrite(output_path, depth)
            else:
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                combined_result = cv2.hconcat([raw_image, split_region, depth])
                
                cv2.imwrite(output_path, combined_result)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--input-size', type=int, default=518)

    parser.add_argument('--img-path', type=str, default='./input_images/test/demo/')
    parser.add_argument('--outdir', type=str, default='./vis_depth')

    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')

    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()
    

    depth_obj = DepthAnythingRunner(args)
    depth_obj.process_and_save_image()