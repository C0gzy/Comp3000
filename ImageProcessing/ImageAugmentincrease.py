# Increase the number of images in the dataset by augmenting the existing images

import os
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm
import asyncio
import random
import shutil
import concurrent.futures

os.makedirs("../CoralDataSetAugmented/train/CORAL", exist_ok=True)
os.makedirs("../CoralDataSetAugmented/train/CORAL_BL", exist_ok=True)
os.makedirs("../CoralDataSetAugmented/val/CORAL", exist_ok=True)
os.makedirs("../CoralDataSetAugmented/val/CORAL_BL", exist_ok=True)
os.makedirs("../CoralDataSetAugmented/test/CORAL", exist_ok=True)
os.makedirs("../CoralDataSetAugmented/test/CORAL_BL", exist_ok=True)

def augment_images(image_dir, output_dir):
    shutil.copytree(image_dir, output_dir, dirs_exist_ok=True)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get a list of all image files in the input directory with jpg, jpeg, or png extensions
    image_files = []
    for file_path in image_dir.iterdir():
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_files.append(file_path)

    for path in tqdm(image_files, desc=f"Augmenting images in {image_dir}", unit="file"):
        with Image.open(path) as img:
            img = img.convert("RGB")

            stem = path.stem
            base_ext = ".png"

            flipped_lr = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_lr.save(output_dir / f"{stem}_flipped_left_right{base_ext}")

            rotated_15 = img.rotate(2)
            rotated_15.save(output_dir / f"{stem}_rotated_2{base_ext}")

            rotated_minus15 = img.rotate(-2)
            rotated_minus15.save(output_dir / f"{stem}_rotated_minus2{base_ext}")

            flipped_15 = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(2)
            flipped_15.save(output_dir / f"{stem}_flipped_left_right_rotated_2{base_ext}")

            flipped_minus15 = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(-2)
            flipped_minus15.save(output_dir / f"{stem}_flipped_left_right_rotated_minus2{base_ext}")

            # Color jitter using PIL's ImageEnhance utilities
            brightness_factor = random.uniform(0.5, 2)
            contrast_factor = random.uniform(0.5, 2)
            color_factor = random.uniform(0.5, 2)
            sharpness_factor = random.uniform(0.5, 2)

            colour_jitter = ImageEnhance.Brightness(img).enhance(brightness_factor)
            colour_jitter = ImageEnhance.Contrast(colour_jitter).enhance(contrast_factor)
            colour_jitter = ImageEnhance.Color(colour_jitter).enhance(color_factor)
            colour_jitter = ImageEnhance.Sharpness(colour_jitter).enhance(sharpness_factor)
            colour_jitter.save(output_dir / f"{stem}_colour_jitter{base_ext}")

            # Zoom (scale up to 254px max, then crop back to original size to remove edges)
            max_zoom = min(254 / img.width, 1.5)
            zoom_factor = random.uniform(1.2, max_zoom)
            zoomed_width = int(img.width * zoom_factor)
            zoomed_height = int(img.height * zoom_factor)
            zoomed = img.resize((zoomed_width, zoomed_height), Image.LANCZOS)

            left = (zoomed_width - img.width) // 2
            top = (zoomed_height - img.height) // 2
            right = left + img.width
            bottom = top + img.height
            zoomed_cropped = zoomed.crop((left, top, right, bottom))
            zoomed_cropped.save(output_dir / f"{stem}_zoomed{base_ext}")



            #flipped_tb = img.transpose(Image.FLIP_TOP_BOTTOM)
            #flipped_tb.save(output_dir / f"{stem}_flipped_top_bottom{base_ext}")

    print(
        f"Augmented {len(image_files)} images in {image_dir}. "
        f"Total files in output: {len(list(output_dir.glob('*')))}"
    )

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(augment_images, "../CoralDataSet/train/CORAL", "../CoralDataSetAugmented/train/CORAL")
        executor.submit(augment_images, "../CoralDataSet/train/CORAL_BL", "../CoralDataSetAugmented/train/CORAL_BL")
        executor.submit(augment_images, "../CoralDataSet/val/CORAL", "../CoralDataSetAugmented/val/CORAL")
        executor.submit(augment_images, "../CoralDataSet/val/CORAL_BL", "../CoralDataSetAugmented/val/CORAL_BL")
        executor.submit(augment_images, "../CoralDataSet/test/CORAL", "../CoralDataSetAugmented/test/CORAL")
        executor.submit(augment_images, "../CoralDataSet/test/CORAL_BL", "../CoralDataSetAugmented/test/CORAL_BL")