from PIL import Image
import glob

imgs = glob.glob('data/test/frames/*.png')
print(f'Found {len(imgs)} frames')
if imgs:
    img = Image.open(imgs[0])
    print(f'Frame dimensions: {img.width}x{img.height}')
