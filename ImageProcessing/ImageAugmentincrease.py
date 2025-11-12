# Increase the number of images in the dataset by augmenting the existing images

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


import shutil

os.makedirs("../CoralDataSetAugmented/train/CORAL", exist_ok=True)
os.makedirs("../CoralDataSetAugmented/train/CORAL_BL", exist_ok=True)

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

            flipped_tb = img.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_tb.save(output_dir / f"{stem}_flipped_top_bottom{base_ext}")

    print(
        f"Augmented {len(image_files)} images in {image_dir}. "
        f"Total files in output: {len(list(output_dir.glob('*')))}"
    )


augment_images("../CoralDataSet/train/CORAL", "../CoralDataSetAugmented/train/CORAL")
augment_images("../CoralDataSet/train/CORAL_BL", "../CoralDataSetAugmented/train/CORAL_BL")