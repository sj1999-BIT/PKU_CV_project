import argparse
import os
import json
import torch
import sys

rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)

from yolo_utils import cxcywh_to_xyxy
from pathlib import Path
from RELTR import cuda_model
from tqdm import tqdm



def generatePredYoloData(outputs):

    yolo_rel_data_output = []
    yolo_obj_data_output = []

    # keep only predictions with 0.+ confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]  # shape (200, 51)
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]  # shape (200, 151)
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]  # shape (200, 151)
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))  # shape (200,)

    # we no need this step as we aim to generate data.
    # # convert boxes from [0; 1] to image scales
    # sub_bboxes_scaled = rescale_bboxes_cxcywh(outputs['sub_boxes'][0, keep], pil_image.size)
    # obj_bboxes_scaled = rescale_bboxes_cxcywh(outputs['obj_boxes'][0, keep], pil_image.size)

    topk = 10
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]

    # confidence scores for relationships, subjects, objects
    # Multiply all three confidence scores together
    # -scores means sort in descending order (highest confidence first)
    # [:topk] takes the top 10 highest confidence predictions
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    # From all valid predictions, keep only the top 10 with highest combined confidence
    keep_queries = keep_queries[indices]

    for idx, s_box, o_box in \
            zip(keep_queries, outputs['sub_boxes'][0, keep], outputs['obj_boxes'][0, keep]):

        # first label N/A and background is excluded
        subject_label = probas_sub[idx].argmax()-1
        object_label = probas_obj[idx].argmax()-1
        predicate_label = probas[idx].argmax()-1

        # we now generate the respectively boxes

        # cx1, cy1, w1, h1 = xyxy_cxcywh(s_box[0].item(), s_box[1].item(), s_box[2].item(), s_box[3].item())
        # cx2, cy2, w2, h2 = xyxy_cxcywh(o_box[0].item(), o_box[1].item(), o_box[2].item(), o_box[3].item())


        yolo_obj_data_output.append([subject_label.item(), s_box[0].item(),s_box[1].item(), s_box[2].item(), s_box[3].item()])
        yolo_obj_data_output.append([object_label.item(), o_box[0].item(), o_box[1].item(), o_box[2].item(), o_box[3].item()])

        yolo_rel_data_output.append([len(yolo_obj_data_output)-1,
                                     len(yolo_obj_data_output)-2,
                                     predicate_label.item()])

    return yolo_rel_data_output, yolo_obj_data_output


def ReltrDataAnnotation(dataFilePath):
    """
    Aim of this function: given a folder named Data:
            Data
            └── images #contains unlabelled images
            └── obj_labels #empty
    Each imagefilepath is passed to function model_inference to obtain outputs.
    """

    # Convert to Path object for easier path manipulation
    data_path = Path(dataFilePath)
    images_path = data_path / "images"
    rel_labels_path = data_path / "rel_labels"
    obj_labels_path = data_path / "obj_labels"

    # Validate folder structure
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found at {dataFilePath}")
    if not images_path.exists():
        raise FileNotFoundError(f"Images folder not found at {images_path}")
    if not rel_labels_path.exists():
        os.makedirs(rel_labels_path)
        print(f"Created rel_labels directory at {rel_labels_path}")
    if not obj_labels_path.exists():
        os.makedirs(obj_labels_path)
        print(f"Created obj_labels directory at {obj_labels_path}")

    # Get list of image files
    image_files = [f for f in images_path.glob("*")
                   if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]

    if not image_files:
        raise ValueError(f"No valid image files found in {images_path}")

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Get model predictions
            predictions = cuda_model.model_inference(str(image_file))

            yolo_reL_tensor_file, yolo_obj_tensor_file = generatePredYoloData(predictions)

            # Create annotation filename (same name as image but .txt extension)
            annotation_file = rel_labels_path / f"{image_file.stem}.txt"

            with open(annotation_file, 'w') as f:
                for yolo_tensor in yolo_reL_tensor_file:
                    for tensor in yolo_tensor:
                        f.write(f"{tensor.item()} ")
                    f.write("\n")

            annotation_file = obj_labels_path / f"{image_file.stem}.txt"
            with open(annotation_file, 'w') as f:
                for yolo_tensor in yolo_obj_tensor_file:
                    for tensor in yolo_tensor:
                        f.write(f"{tensor.item()} ")
                    f.write("\n")

        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
            # Print the contents of the tensor to understand what went wrong
            if 'yolo_reL_tensor_file' in locals():
                print("Contents of yolo_reL_tensor_file:", yolo_reL_tensor_file)
            if 'yolo_obj_tensor_file' in locals():
                print("Contents of yolo_obj_tensor_file:", yolo_obj_tensor_file)
            continue


    print("Annotation process completed")
    return True

def FormattedReltrDataAnnotation(dataFilePath):
    """
    Aim of this function: given a folder named Data:
        dataset
           ├── images/               # All image files
           ├── obj_labels/          # YOLO format object annotations
           ├── pred_labels/          # Relationship triplet annotations
    """

    # Convert to Path object for easier path manipulation
    data_path = Path(dataFilePath)
    images_path = data_path / "images"
    rel_labels_path = data_path / "pred_labels"
    obj_labels_path = data_path / "obj_labels"

    # Validate folder structure
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found at {dataFilePath}")
    if not images_path.exists():
        raise FileNotFoundError(f"Images folder not found at {images_path}")
    if not rel_labels_path.exists():
        os.makedirs(rel_labels_path)
        print(f"Created rel_labels directory at {rel_labels_path}")
    if not obj_labels_path.exists():
        os.makedirs(obj_labels_path)
        print(f"Created obj_labels directory at {obj_labels_path}")

    # Get list of image files
    image_files = [f for f in images_path.glob("*")
                   if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]

    if not image_files:
        raise ValueError(f"No valid image files found in {images_path}")

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Get model predictions
            predictions = cuda_model.model_inference(str(image_file))

            yolo_reL_tensor_file, yolo_obj_tensor_file = generatePredYoloData(predictions)

            # Create annotation filename (same name as image but .txt extension)
            annotation_file = rel_labels_path / f"{image_file.stem}.txt"

            with open(annotation_file, 'w') as f:
                for yolo_tensor in yolo_reL_tensor_file:
                    for tensor in yolo_tensor:
                        f.write(f"{tensor} ")
                    f.write("\n")

            annotation_file = obj_labels_path / f"{image_file.stem}.txt"
            with open(annotation_file, 'w') as f:
                for yolo_tensor in yolo_obj_tensor_file:
                    for tensor in yolo_tensor:
                        f.write(f"{tensor} ")
                    f.write("\n")

        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
            # Print the contents of the tensor to understand what went wrong
            if 'yolo_reL_tensor_file' in locals():
                print("Contents of yolo_reL_tensor_file:", yolo_reL_tensor_file)
            if 'yolo_obj_tensor_file' in locals():
                print("Contents of yolo_obj_tensor_file:", yolo_obj_tensor_file)
            continue


    print("Annotation process completed")
    return True


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Process data with ReltrDataAnnotation')
    # parser.add_argument('--yoloGen_datapath', type=str, help='Path to the data directory')
    #
    # args = parser.parse_args()
    #
    # print(f"parser.parse_args() {parser.parse_args()}")

    yoloGen_datapath = "../../autodl-tmp/yolo_25K/"
    FormattedReltrDataAnnotation(yoloGen_datapath)