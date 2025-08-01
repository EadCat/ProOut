from PIL import Image


def pil_maximum_size(img: Image.Image, max_edge_size=1536):
    w, h = img.size
    if h > max_edge_size or w > max_edge_size:
        smaller_edge = round(max_edge_size / max(h, w) * min(h, w))
        if h >= w:
            img = img.resize((smaller_edge, max_edge_size), Image.NEAREST)
        else:
            img = img.resize((max_edge_size, smaller_edge), Image.NEAREST)
    return img


def pil_minimum_size(img: Image.Image, min_edge_size=512):
    w, h = img.size
    if h < min_edge_size or w < min_edge_size:
        larger_edge = round(min_edge_size / min(h, w) * max(h, w))
        # PIL.Image.resize([Width, Height])
        if h >= w:
            img = img.resize((min_edge_size, larger_edge), Image.BICUBIC)
        else:
            img = img.resize((larger_edge, min_edge_size), Image.BICUBIC)
    return img

