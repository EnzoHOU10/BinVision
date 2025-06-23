from PIL import Image
import os

def extract_features(image_path):
    img = Image.open(image_path)
    width, height = img.size
    filesize_kb = os.path.getsize(image_path) / 1024
    pixels = list(img.getdata())
    r, g, b = zip(*pixels) if img.mode == 'RGB' else (0, 0, 0)
    avg_color = (int(sum(r) / len(r)), int(sum(g) / len(g)), int(sum(b) / len(b)))
    return width, height, filesize_kb, *avg_color
