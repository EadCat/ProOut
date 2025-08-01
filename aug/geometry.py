import random
from math import floor, ceil
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

def mask_creation(height: int, width: int, exp_h: int, exp_w: int) -> torch.Tensor:
    # Create a mask with the same shape as the image
    # Fill the mask with 1s to be generated, and 0s for known pixels
    mask = torch.full(size=[1, height, width], fill_value=1., dtype=torch.float32)
    mask[:, exp_h: -exp_h, exp_w: -exp_w] = 0.

    return mask  # [1, H, W]


def canvas_sweep(image: torch.Tensor, sweep_h: int, sweep_w: int, colors: list) -> torch.Tensor:
    """
    image: [*, C, H, W]
    """
    if isinstance(colors, (int, float)): colors = [colors] * 3
    assert len(colors) == 3 or len(colors) == 1, f"Check the number of colors: {colors}"
    if len(colors) == 1: colors = colors * 3  # [N] -> [N, N, N]
    
    copy_img = image.clone()
    # Sweep the canvas with designated colors
    for chn, color in enumerate(colors):
        copy_img[[chn], :sweep_h] = color
        copy_img[[chn], -sweep_h:] = color
        copy_img[[chn], :, :sweep_w] = color
        copy_img[[chn], :, -sweep_w:] = color

    return copy_img


# def get_random_outline_window_coordinates(image: torch.Tensor, window: int = 512) -> tuple[int, int]:
#     *_, h, w = F.get_dimensions(image)
#     # Choose the direction
#     direction = random.choice(['up', 'down', 'left', 'right'])
#     if direction == 'up':
#         y = 0
#         x = random.randint(0, w - window)
#     elif direction == 'down':
#         y = h - window
#         x = random.randint(0, w - window)
#     elif direction == 'left':
#         y = random.randint(0, h - window)
#         x = 0
#     elif direction == 'right':
#         y = random.randint(0, h - window)
#         x = w - window
#     return x, y


def get_random_outline_sticked_window_coordinates(
    h: int,
    w: int,
    remove_h: int,
    remove_w: int,
    window: int = 512,  # [qK, K]
    spare_pixels: int = 4,  # u
    ) -> tuple[int, int]:
    direction = random.choice(['up', 'down', 'left', 'right'])
    h_spare = spare_pixels if remove_h >= window else 0
    w_spare = spare_pixels if remove_w >= window else 0
    
    if direction == 'up':
        y = random.randint(max(0, remove_h - window) + h_spare, remove_h - h_spare)
        x = random.randint(max(0, remove_w - window) + w_spare, w - max(remove_w, window) - w_spare)
    elif direction == 'down':
        y = random.randint(h - (remove_h + window) + h_spare, h - max(remove_h, window) - h_spare)
        x = random.randint(max(0, remove_w - window) + w_spare, w - max(remove_w, window) - w_spare)
    elif direction == 'left':
        y = random.randint(max(0, remove_h - window) + h_spare, h - max(remove_h, window) - h_spare)
        x = random.randint(max(0, remove_w - window) + w_spare, remove_w - w_spare)
    elif direction == 'right':
        y = random.randint(max(0, remove_h - window) + h_spare, h - max(remove_h, window) - h_spare)
        x = random.randint(w - (remove_w + window) + w_spare, w - max(remove_w, window) - w_spare)
    else:
        raise ValueError("Illegal Direction for cropping input")

    return x, y

def random_outline_attach(original_image, # I^e
                          canvas,  # Initial I_C
                          mask, # Initial M_global
                          remove_h, 
                          remove_w, 
                          max_attach_num=32, 
                          attach_window=512,   # K
                          spare_pixels=4,      # u
                          min_crop_ratio=0.9,  # q <= 1
                          escape_threshold=0.15
                          ):
    *_, h, w = F.get_dimensions(original_image)
    # Randomly choose the number of windows to attach
    attach_num = random.randint(0, max_attach_num)
    # Attach the windows to the canvas [source (canvas) -> target (original_image)]
    for _ in range(attach_num):
        # Randomly choose the size of the source window to attach  [qK, K]
        attach_window_edge = random.randint(int(attach_window * min_crop_ratio), attach_window)
        # Randomly choose the position within the target
        x, y = get_random_outline_sticked_window_coordinates(h, w, remove_h, remove_w, attach_window_edge, spare_pixels)
        # Attach
        canvas[..., y: y+attach_window_edge, x: x+attach_window_edge] = original_image[..., y: y+attach_window_edge, x: x+attach_window_edge]
        mask[..., y: y+attach_window_edge, x: x+attach_window_edge] = 0.
        if torch.mean(mask) <= escape_threshold: break  # Manipulate valid ratio of the overall mask
    return canvas, mask
    
def get_sticked_window_coordinates(
    image: torch.Tensor,
    mask: torch.Tensor,
    model_window: int = 512,
    crop: int = 512,
    escape_threshold: float = 0.05
    ) -> tuple[int, int]:
    *_, h, w = F.get_dimensions(image)
    x_limit = w - crop
    y_limit = h - crop
    while True:
        # Randomly choose the crop window
        x, y = random.randint(0, x_limit), random.randint(0, y_limit)
        # Judge the crop window is valid or not
        crop_mask = mask[..., y: y + model_window, x: x + model_window]
        if torch.mean(crop_mask) > escape_threshold: break
    return x, y
    


def hint_pad(image: torch.Tensor, window_size: int=512, fill: list=[0, 0, 0]):
    """
    Pad Hint to a square input.
    h : w = rh : rw; w * rh = h * rw
    1) rw = w * rh / h
    2) rh = h * rw / w
    """
    *_, h, w = image.shape
    if isinstance(fill, (int, float)): fill = [fill] * 3
    assert len(fill) == 3 or len(fill) == 1, f"Check the number of colors: {fill}"
    if len(fill) == 1: fill = fill * 3  # [N] -> [N, N, N]
    
    # 1) Downsampling
    if h > w:
        rh = window_size
        rw = int(round(w * window_size / h))
    else:
        rw = window_size
        rh = int(round(h * window_size / w))
    image = F.resize(image, [rh, rw], antialias=True,
                        interpolation=InterpolationMode.BILINEAR)
    # 2) Padding
    pad_h = window_size - rh
    pad_w = window_size - rw
    top, bottom = floor(pad_h / 2), ceil(pad_h / 2)
    left, right = floor(pad_w / 2), ceil(pad_w / 2)
    image = torch.stack([F.pad(img, [left, top, right, bottom], val) for img, val
                        in zip(image, fill)], dim=0)
    return image
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    