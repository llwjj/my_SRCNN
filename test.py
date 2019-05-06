import argparse
import numpy as np
from pathlib import Path

from model import get_model
from image_process import imread,imshow,LR_image

def get_args():
    args = argparse.ArgumentParser(description="Super-Resolution_args",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # args.add_argument('--test_dir',type=str,required=True,help='train image dir')
    # args.add_argument('--model_file',type=str,required=True)
    args.add_argument('--test_dir',type=str,default='../dataset_images/Set14',help='image dir')
    args.add_argument('--model_file',type=str,default='./result/weights.027-215.611-24.79650.hdf5')

    args.add_argument('--de_num',type=int,default=4,help='low image')
    args.add_argument('--model',type=str,default='SRCNN')
    args = args.parse_args()
    return args

def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)

def main():
    args = get_args()
    model = get_model(args.model)
    model.load_weights(args.model_file)

    image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
    image_paths = [p for p in Path(args.test_dir).glob("**/*") if p.suffix.lower() in image_suffixes]

    for p in image_paths:
        image = imread(str(p))
        h, w, _ = image.shape
       
        out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
        noise_image = LR_image(image,args.de_num)
        pred = model.predict(np.expand_dims(noise_image, 0))
        denoised_image = get_image(pred[0])

        out_image[:, :w] = image
        out_image[:, w:w * 2] = noise_image
        out_image[:, w * 2:] = denoised_image

        if imshow(out_image) == 113:
            return 0

if __name__ == "__main__":
    main()