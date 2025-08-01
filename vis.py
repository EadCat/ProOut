import os
import glob
import numpy as np
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from aug.safety import pil_maximum_size, pil_minimum_size
from aug.geometry import (
    mask_creation,
    canvas_sweep,
    random_outline_attach,
    get_sticked_window_coordinates,
    hint_pad
    )
from utils.tensorIO import tensor_save

import torch


if __name__ == '__main__':
    imgroot = r'./images/'
    imglist = sorted(glob.glob(os.path.join(imgroot, '*')))

    dstroot = r'./results'
    os.makedirs(dstroot, exist_ok=True)

    window = 512
    max_damper = int(round(window * 4))
    min_safety = 1280
    remove_ratio = 0.5

    fetch_num = 32

    for imgdir in tqdm(imglist):
        name = os.path.splitext(os.path.basename(imgdir))[0]  # 00000.jpg -> 00000
        image = Image.open(imgdir).convert('RGB')

        # Resize the image to be cropped safely
        image = np.array(pil_minimum_size(pil_maximum_size(image, max_damper), min_safety), 
                         dtype=np.float32)
        # [0, 255] -> [0, 1]
        image = torch.from_numpy(rearrange(image, 'h w c -> c h w')) / 255

        *_, h, w = image.shape
        # Calculate the size of the removed area
        remove_h = int(round(h * remove_ratio // 2))
        remove_w = int(round(w * remove_ratio // 2))

        # Remove the known pixels to "GRAY"
        canvas = canvas_sweep(image, remove_h, remove_w, [0.5])
        # Create a mask with the same size as the image
        mask = mask_creation(h, w, remove_h, remove_w)

        # Augmentation Algorithms Start:
        canvas, mask = random_outline_attach(image,
                                             canvas,
                                             mask,
                                             remove_h, remove_w,
                                             fetch_num, window)

        x, y = get_sticked_window_coordinates(canvas, mask, window)
        crop_gt = image[..., y:y + window, x:x + window]
        crop = canvas[..., y:y + window, x:x + window]
        crop_mask = mask[..., y:y + window, x:x + window]

        hint_targeting = torch.zeros_like(mask)
        hint_targeting[..., y:y + window, x:x + window] = 1

        hint = hint_pad(canvas, window, [0.5])
        hint_mask = hint_pad(mask, window)
        hint_targeting = hint_pad(hint_targeting, window)

        export_dir = os.path.join(dstroot, f'{name}')

        tensor_save(crop_gt, export_dir + '-Local-GT')
        tensor_save(crop, export_dir + '-Local-Input')
        tensor_save(crop_mask, export_dir + '-Local-Mask')
        tensor_save(hint, export_dir + '-Global-Hint')
        tensor_save(hint_mask, export_dir + '-Global-Mask')
        tensor_save(hint_targeting, export_dir + '-Global-Highlight')
