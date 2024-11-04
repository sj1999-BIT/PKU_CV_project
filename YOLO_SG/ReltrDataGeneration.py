import os
import json

from pathlib import Path
from RELTR import cuda_model


def ReltrDataAnnotation(dataFilePath):
    """
    Aim of this function: given a folder named Data:
            Data
            └── images #contains unlabelled images
            └── labels #empty
    Each imagefilepath is passed to function model_inference to obtain outputs.
    """

    # Convert to Path object for easier path manipulation
    data_path = Path(dataFilePath)
    images_path = data_path / "images"
    labels_path = data_path / "labels"

    # Validate folder structure
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found at {dataFilePath}")
    if not images_path.exists():
        raise FileNotFoundError(f"Images folder not found at {images_path}")
    if not labels_path.exists():
        os.makedirs(labels_path)
        print(f"Created labels directory at {labels_path}")

    # Get list of image files
    image_files = [f for f in images_path.glob("*")
                   if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]

    if not image_files:
        raise ValueError(f"No valid image files found in {images_path}")

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for image_file in image_files:
        try:
            # Get model predictions
            predictions = cuda_model.model_inference(str(image_file))

            print(predictions)

            # Create annotation filename (same name as image but .json extension)
            annotation_file = labels_path / f"{image_file.stem}.txt"


        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
            continue

    print("Annotation process completed")
    return True


if __name__ == '__main__':
    data_filepath = "./coco_dataset/sample_img_data"
    ReltrDataAnnotation(data_filepath)