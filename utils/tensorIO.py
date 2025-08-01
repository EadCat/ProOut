import numpy as np
from einops import rearrange
import cv2
import torch


def numpy2tensor(image: np.ndarray, dtype: torch.dtype=torch.float32, device=torch.device('cpu')) -> torch.Tensor:
    image = torch.from_numpy(rearrange(image, 'h w c -> c h w')).to(dtype=dtype, device=device)
    return image


def tensor2numpy(tensor: torch.Tensor, dtype: np.dtype=np.float32) -> np.ndarray:
    if tensor.ndim == 4: tensor = tensor.squeeze(0)
    np_tensor = np.array(tensor.cpu(), dtype=dtype)
    np_tensor = rearrange(np_tensor, 'c h w -> h w c')
    return np_tensor


def tensor_save(tensor: torch.Tensor, path: str, ext="png"):
    if tensor.ndim == 4 and tensor.shape[0] > 1:
        for i, t in enumerate(tensor):
            cv2.imwrite(path + f"_{str(i).zfill(4)}.{ext}", tensor2numpy(t * 255, dtype=np.uint8)[..., ::-1])
    else:
        cv2.imwrite(path + f".{ext}", tensor2numpy(tensor * 255, dtype=np.uint8)[..., ::-1])

