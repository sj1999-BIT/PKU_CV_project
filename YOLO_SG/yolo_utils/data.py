import json
import math
import os
import time
import sys
import shutil

import cv2
import yaml
import random

from collections import defaultdict

from tqdm import tqdm
from yolo_utils.label_utils import *

rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)


# only print if code is testing
# def test_log(log_stmt, is_log_printing):
#     if is_log_printing:
#         print(log_stmt)

# function generate true based on input percentage
def random_true(true_chance=10):
    # Generate a random number between 0 and 1
    random_number = random.random()

    # Check if the random number is less than 0.1 (10% probability)
    if random_number < true_chance / 100:
        return True
    else:
        return False


def create_folder(folder_path, is_log_printing=False):
    # Create the output folder if it doesn't exist
    if not os.path.exists(folder_path):
        test_log(f"new folder created {folder_path}", is_log_printing)
        os.makedirs(folder_path)


def create_file_list(txt_file_path, target_dir, is_log_printing=False):
    img_dir = os.path.join(os.path.dirname(txt_file_path), target_dir)
    # cur_folder = os.path.join(target_dir.split(os.sep)[-2])
    with open(txt_file_path, 'w') as f:
        for file in os.listdir(img_dir):
            input_path = os.path.join("./", target_dir, file) + "\n"
            test_log(f'path writter: {input_path}', is_log_printing)
            f.write(input_path)


def create_yaml_file(data_path, label_dict, yaml_filename="lighthaus_data.yaml", is_log_printing=False):
    config = {
        'path': data_path,
        'train': "train_lighthaus.txt",
        'val': "val_lighthaus.txt",
        'test': "test_lighthaus.txt",
        'names': {i: label for i, label in label_dict.items()}
    }

    yaml_file_path = os.path.join(data_path, yaml_filename)
    with open(yaml_file_path, 'w') as file:
        test_log(f"yaml created with {config} at {yaml_file_path}", is_log_printing)
        yaml.dump(config, file)

    return yaml_file_path


# clears up the folder
def clear_folder(folder_path, is_log_printing=False):
    # Verify if the folder path exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                test_log(f'remove file {file_path}', is_log_printing)
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # Recursively clear files in subfolder
                clear_folder(file_path)
            else:
                test_log(f"Skipping non-file item: {filename}", is_log_printing)
    else:
        test_log("Folder does not exist or is not a directory.", is_log_printing)


def prepare_data_folder(folder_path, is_log_printing=False):
    """
    # folder requires all yolo data should have an arrangement of
#   └── images
#   └── labels
    """
    create_folder(folder_path)
    create_folder(os.path.join(folder_path, "images"))
    create_folder(os.path.join(folder_path, "labels"))


def prepare_yolo_data_folder(folder_path, is_log_printing=False):
    """
    folder requires all yolo data should have an arrangement of
    └── train
    └── val
    └── test
        └── images
        └── labels
    :param folder_path: location for the folder
    :param is_log_printing: show logs
    :return:
    """
    # Create the output folder if it doesn't exist
    create_folder(folder_path)
    prepare_data_folder(os.path.join(folder_path, "train"))
    prepare_data_folder(os.path.join(folder_path, "val"))
    prepare_data_folder(os.path.join(folder_path, "test"))


def prepare_error_labelled_img_folder(folder_path, target_label, is_log_printing=False):
    """
    folder contains all labelled images for invalid images
    └── invalid labelled images
        └── target_label
            └── false positives
            └── false negatives
    :param folder_path:
    :param is_log_printing:
    :return:
    """
    invalid_img_folder_path = os.path.join(folder_path, "invalid_labelled_images")

    create_folder(invalid_img_folder_path, is_log_printing)

    target_label_invalid_img_folder_path = os.path.join(invalid_img_folder_path, target_label)
    create_folder(target_label_invalid_img_folder_path, is_log_printing)

    create_folder(os.path.join(target_label_invalid_img_folder_path, "false_positives"), is_log_printing)
    create_folder(os.path.join(target_label_invalid_img_folder_path, "false_negatives"), is_log_printing)


def prepare_true_labelled_img_folder(folder_path, target_label, is_log_printing=False):
    """
    folder contains all labelled images for invalid images
    └── invalid labelled images
        └── target_label
            └── true_positives
            └── true_negatives
    :param folder_path:
    :param is_log_printing:
    :return:
    """
    invalid_img_folder_path = os.path.join(folder_path, "valid_labelled_images")

    create_folder(invalid_img_folder_path, is_log_printing)

    target_label_invalid_img_folder_path = os.path.join(invalid_img_folder_path, target_label)
    create_folder(target_label_invalid_img_folder_path, is_log_printing)

    create_folder(os.path.join(target_label_invalid_img_folder_path, "true_positives"), is_log_printing)
    create_folder(os.path.join(target_label_invalid_img_folder_path, "true_negatives"), is_log_printing)


def copy_img_folder(source_folder, destination_folder, is_log_printing=False, remove_old_img=False):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of all files in the source folder
    files = os.listdir(source_folder)

    # Filter only image files (you can extend the list of extensions if needed)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Copy each image file to the destination folder
    for image_file in image_files:
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        if remove_old_img:
            shutil.move(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)
        test_log(f"Copied: {image_file}", is_log_printing=is_log_printing)


def get_all_label_filename(folder_path):
    """
    Give the main data folder, return a list of label filenames
    :param folder_path: folder should contain a subfolder named labels
    :return: a list of strings
    """
    label_folder = os.path.join(folder_path, "labels")

    # get all the labels from both files
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    return label_files


def generate_empty_label_files(folder_path):
    label_folder = os.path.join(folder_path, "labels")
    img_folder = os.path.join(folder_path, "images")

    # Iterate through all files in the folder
    for filename in os.listdir(img_folder):
        # Create an empty text file with the same name
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_filepath = os.path.join(label_folder, txt_filename)
        if not os.path.exists(txt_filepath):
            with open(txt_filepath, 'w') as txt_file:
                pass  # Writing nothing to create an empty file


def transfer_yolo_data(img_file, folder_path, out_folder_path, is_log_printing=False):
    label_file = os.path.splitext(img_file)[0] + ".txt"
    img_file = os.path.splitext(img_file)[0] + ".jpg"

    img_folder = os.path.join(folder_path, "images")
    label_folder = os.path.join(folder_path, "labels")

    # original data path
    cur_img_path = os.path.join(img_folder, img_file)
    cur_label_path = os.path.join(label_folder, label_file)

    # new data path
    new_img_path = os.path.join(out_folder_path, "images", img_file)
    new_label_path = os.path.join(out_folder_path, "labels", label_file)

    # copy data over
    shutil.copy(cur_img_path, new_img_path)
    if not os.path.exists(cur_label_path):
        # no original labels, then just create a empty file
        with open(new_label_path, 'w') as f:
            pass
    else:
        # copy over the label txt file
        shutil.copy(cur_label_path, new_label_path)


def separate_data(folder_path, output_folder_path, val_percent=10.0, test_percent=10.0, is_log_printing=False):
    """
    Given a folder with images and labels, copy and separate them into train, val, test
    :param folder_path: folder containing images and label
    """
    clear_folder(output_folder_path)
    prepare_yolo_data_folder(output_folder_path)

    img_folder = os.path.join(folder_path, "images")
    label_folder = os.path.join(folder_path, "labels")

    # List all label files in the source folder
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    # List all label files in the source folder
    img_files = [f for f in os.listdir(img_folder)]

    val_count = 0
    is_val_empty = True
    val_out_path = os.path.join(output_folder_path, "val")

    test_count = 0
    is_test_empty = True
    test_out_path = os.path.join(output_folder_path, "test")

    # allocate data to validation testset and test dataset until both folder are not empty
    while is_val_empty or is_test_empty:
        for img_file in tqdm(img_files, desc="allocating images to valid and test"):
            if random_true(int(val_percent)):
                transfer_yolo_data(img_file, folder_path, val_out_path)
                val_count += 1
                is_val_empty = False
            elif random_true(int(test_percent)):
                transfer_yolo_data(img_file, folder_path, test_out_path)
                test_count += 1
                is_test_empty = False

    # transfer the rest to train folder
    for img_file in tqdm(img_files, desc="copying images to valid and test"):
        if img_file in os.listdir(val_out_path) or img_file in os.listdir(test_out_path):
            continue
        out_folder_path = os.path.join(output_folder_path, "train")
        transfer_yolo_data(img_file, folder_path, out_folder_path)

    test_log(
        f'total data separated: train has {len(img_files) - val_count - test_count}, val has {val_count}, test has {test_count}',
        is_log_printing)


def prep_txt_file_for_yolo(target_data_dir, is_log_printing=False):
    """
    all yolo train data should have an arrangement of
    └── datasets
        └── train
        └── val
        └── test
            └── images
            └── labels
    can just perform txt file creation given the main data dir
    :param target_data_dir:
    :return:
    """
    # train_dir = os.path.join(target_data_dir, "train")
    # val_dir = os.path.join(target_data_dir, "val")
    # test_dir = os.path.join(target_data_dir, "test")

    train_dir = os.path.join(target_data_dir, "train")
    val_dir = os.path.join(target_data_dir, "val")
    test_dir = os.path.join(target_data_dir, "test")

    create_file_list(os.path.join(target_data_dir, "train_lighthaus.txt"), os.path.join("train", "images"),
                     is_log_printing)
    create_file_list(os.path.join(target_data_dir, "val_lighthaus.txt"), os.path.join("val", "images"), is_log_printing)
    create_file_list(os.path.join(target_data_dir, "test_lighthaus.txt"), os.path.join("test", "images"),
                     is_log_printing)


def prepare_yolo_data_folder(folder_path, is_log_printing=False):
    """
    folder requires all yolo data should have an arrangement of
    └── train
    └── val
    └── test
        └── images
        └── labels
    :param folder_path: location for the folder
    :param is_log_printing: show logs
    :return:
    """
    # Create the output folder if it doesn't exist
    create_folder(folder_path)
    prepare_data_folder(os.path.join(folder_path, "train"))
    prepare_data_folder(os.path.join(folder_path, "val"))
    prepare_data_folder(os.path.join(folder_path, "test"))


def full_yolo_data_prep_pipeline(label_dict, original_data_path, trainable_data_path):
    if os.path.exists(trainable_data_path):
        print(f"training data already exist at {trainable_data_path}")
        return trainable_data_path

    prepare_yolo_data_folder(trainable_data_path, is_log_printing=False)
    # separate the data
    separate_data(original_data_path, trainable_data_path, val_percent=10.0, test_percent=10.0, is_log_printing=True)
    # create the required txt_files for yolo training
    prep_txt_file_for_yolo(trainable_data_path, is_log_printing=False)
    # Get the absolute path
    absolute_train_data_path = os.path.abspath(trainable_data_path)
    # create yaml file for training, return the yaml file path as its needed for training
    yaml_file_path = create_yaml_file(absolute_train_data_path, label_dict, is_log_printing=True)
    return yaml_file_path


def create_balance_data(original_data_path, label_count, new_data_path=None, min_val=-1):
    """
    Creates a balanced dataset by intelligently sampling from the original dataset.
    Handles heavily skewed data by using inverse frequency weighting and target quotas.

    Args:
        original_data_path (str): Path to original YOLO dataset
        label_count (list): Current count of each label class
        new_data_path (str, optional): Path for balanced dataset output
        min_val (int, optional): Target minimum count per class. If -1, uses mean

    The algorithm:
    1. Sets target quotas for each label based on min_val
    2. Tracks remaining quota for each label
    3. Calculates selection probability based on:
       - How far each label is from its quota
       - Inverse of current label frequency
    4. Prioritizes underrepresented labels

    Returns:
        str: Path to balanced dataset
    """
    if new_data_path is None:
        new_data_path = os.path.join(os.path.dirname(original_data_path),
                                     f"balanced_{os.path.basename(original_data_path)}")

    if new_data_path is not None and os.path.exists(new_data_path):
        print(f"cleaning up folder {new_data_path} for new data")
        clear_folder(new_data_path)

    prepare_data_folder(new_data_path)

    # Set minimum target value
    if min_val < 0:
        min_val = int(sum(label_count) / len(label_count))

    # Initialize quota tracking
    # target_quota = {i: min_val for i in range(len(label_count))}
    current_count = {i: 0 for i in range(len(label_count))}

    # device a weight, smaller label count, increase weight
    weight_dict = {}
    for i in range(len(label_count)):
        if label_count[i] == 0:
            continue
        weight_dict[i] = min_val / label_count[i]

    weight_sum = sum(list(weight_dict.values())) / len(weight_dict.keys())

    img_folder = os.path.join(original_data_path, "images")
    label_folder = os.path.join(original_data_path, "labels")
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    for label_filename in tqdm(label_files, desc="Balancing dataset"):
        cur_label_path = os.path.join(label_folder, label_filename)
        label_list = read_labels_from_file(cur_label_path, have_confident=False)

        # Calculate importance score for this image
        image_score = 0
        has_useful_labels = False

        # Count labels in current image
        label_counts_in_image = {}
        for label in label_list:
            cur_index = get_label_index(label)
            if cur_index not in label_counts_in_image:
                label_counts_in_image[cur_index] = 0
            label_counts_in_image[cur_index] += 1

        # Calculate score based on needed labels
        for label_idx, count in label_counts_in_image.items():
            if label_count[label_idx] == 0:
                continue

            image_score += weight_dict[label_idx]
            # Calculate how far we are from target for this label
            # remaining_needed = max(0, target_quota[label_idx] - current_count[label_idx])
            # if remaining_needed > 0:
            # Score based on how much this label is needed
            # label_importance = target_quota[label_idx] / label_count[label_idx]
            # if label_importance > image_score:
            #     image_score = label_importance
            # Additional weight for rare labels
            # rarity_weight = min_val / max(label_count[label_idx], 1)
            # image_score += label_importance * rarity_weight
            has_useful_labels = True

        # Normalize score
        if has_useful_labels:
            if image_score > random.random():
                # Copy the files
                img_filename = label_filename.split(".")[0] + ".jpg"
                cur_img_path = os.path.join(img_folder, img_filename)
                new_img_path = os.path.join(new_data_path, "images", img_filename)
                new_label_path = os.path.join(new_data_path, "labels", label_filename)

                shutil.copy(cur_img_path, new_img_path)
                shutil.copy(cur_label_path, new_label_path)

                # Update current counts
                for label_idx, count in label_counts_in_image.items():
                    current_count[label_idx] += count

    # Print statistics about the balancing
    print("\nLabel distribution after balancing:")
    for i in range(len(label_count)):
        if label_count[i] > 0:  # Only show labels that existed in original dataset
            print(f"Label {i}: Original: {label_count[i]}, New: {current_count[i]}")

    return new_data_path


def forced_removal_new_data(original_data_path, label_count, new_data_path=None, max_val=-1):
    """
    Idea: forcefully remove labels that are above threshold amount to get more evenly spread model.
    """
    # Set max count value to keep the label as average if not given
    if max_val < 0:
        max_val = int(sum(label_count) / len(label_count))

    # prepare the output data folder
    if new_data_path is None:
        new_data_path = os.path.join(os.path.dirname(original_data_path),
                                     f"balanced_{os.path.basename(original_data_path)}")

    if new_data_path is not None and os.path.exists(new_data_path):
        print(f"cleaning up folder {new_data_path} for new data")
        clear_folder(new_data_path)

    prepare_data_folder(new_data_path)

    # device a list to decide if we keep the label
    keep = [(label_count[i] <= max_val) for i in range(len(label_count))]

    # target src folder
    img_folder = os.path.join(original_data_path, "images")
    label_folder = os.path.join(original_data_path, "labels")

    # get all label files
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    # count the new data number
    count = 0

    # count the new data label
    current_count = {i: 0 for i in range(len(label_count))}

    # for each label file we decide if we keeping it
    for label_filename in tqdm(label_files, desc="Balancing dataset"):
        cur_label_path = os.path.join(label_folder, label_filename)
        label_list = read_labels_from_file(cur_label_path, have_confident=False)

        # we only collect labels we deem necessary
        new_label_list = []
        for label in label_list:
            cur_index = get_label_index(label)
            # skip unwanted label
            if not keep[cur_index]:
                continue
            # keep the label
            new_label_list.append(label)

        # ignore files with no label we need
        if len(new_label_list) == 0:
            continue

        # new data transferred
        count += 1

        # Copy the files
        img_filename = label_filename.split(".")[0] + ".jpg"
        cur_img_path = os.path.join(img_folder, img_filename)
        new_img_path = os.path.join(new_data_path, "images", img_filename)
        new_label_path = os.path.join(new_data_path, "labels", label_filename)

        shutil.copy(cur_img_path, new_img_path)

        with open(new_label_path, 'w') as f:
            for label in new_label_list:
                current_count[label[0]] += 1
                if label_count[label[0]] > 100:
                    print("WTF")
                f.write(f"{int(label[0])} {label[1]} {label[2]} {label[3]} {label[4]}\n")

    print(f"new data count: {count} at {new_data_path}")
    # Print statistics about the balancing
    print("\nLabel distribution after balancing:")
    for i in range(len(label_count)):
        if label_count[i] > 0:  # Only show labels that existed in original dataset
            print(f"Label {i}: Original: {label_count[i]}, New: {current_count[i]}")

    return new_data_path


def convert_predData_to_relObjData(sg_formatted_data_filepath):
    """
    sg_formatted data directory:
    └── images
    └── object labels.txt
    └── predicate labels.txt
    └── object labels
    └── predicate labels

    1. convert predicates formatted labels into object-formatted labels
    2. store it in a new folder called rel_obj_labels
    """
    # make sure all the files are available
    image_folder = os.path.join(sg_formatted_data_filepath, "images")
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Data image folder not found at {image_folder}")

    obj_labels_folder = os.path.join(sg_formatted_data_filepath, "obj_labels")
    if not os.path.exists(obj_labels_folder):
        raise FileNotFoundError(f"Data obj_labels folder not found at {obj_labels_folder}")

    pred_labels_folder = os.path.join(sg_formatted_data_filepath, "pred_labels")
    if not os.path.exists(pred_labels_folder):
        raise FileNotFoundError(f"Data rel_labels folder not found at {pred_labels_folder}")

    obj_labels_filepath = os.path.join(sg_formatted_data_filepath, "obj_labels.txt ")
    if not os.path.exists(obj_labels_filepath):
        raise FileNotFoundError(f"Data obj_labels.txt not found at {obj_labels_filepath}")

    pred_labels_filepath = os.path.join(sg_formatted_data_filepath, "pred_labels.txt")
    if not os.path.exists(pred_labels_filepath):
        raise FileNotFoundError(f"Data pred_labels_filepath folder not found at {pred_labels_filepath}")

    rel_obj_labels_filepath = os.path.join(sg_formatted_data_filepath, "rel_obj_labels")
    if not os.path.exists(rel_obj_labels_filepath):
        create_folder(rel_obj_labels_filepath)
    else:
        # clean up the folder, remove previous labels
        clear_folder(rel_obj_labels_filepath)

    for image_filename in tqdm(os.listdir(image_folder), "labelling images"):

        if not image_filename.endswith(('.jpg', '.jpeg', '.png')):
            continue

        label_filename = image_filename.split('.')[0] + ".txt"

        # get object labels
        obj_label_data = read_labels_from_file(os.path.join(obj_labels_folder, label_filename),
                                               have_confident=False)

        # generate the corresponding rel obj label files
        rel_obj_data = []
        try:
            with open(os.path.join(pred_labels_folder, label_filename), 'r') as pred_file:
                for line in pred_file:
                    parts = line.strip().split()
                    obj_index, sub_index, pred_index = map(int, parts)
                    obj_box = get_label_box(obj_label_data[obj_index])
                    sub_box = get_label_box(obj_label_data[sub_index])
                    pred_box = merge_cxcywh(obj_box, sub_box)

                    rel_obj_data.append([pred_index, pred_box[0], pred_box[1], pred_box[2], pred_box[3]])
        except FileNotFoundError:
            print(f"File not found: {os.path.join(pred_labels_folder, label_filename)}")

        # current rel_obj may have many highly overlapping boxes with very similar boxes

        concise_rel_obj_data = []

        for i, rel_obj_label in enumerate(rel_obj_data):
            is_overlap = False
            for next_rel_obj_label in rel_obj_data[i + 1:]:
                if get_label_index(rel_obj_label) != get_label_index(next_rel_obj_label):
                    continue
                # only calculate iou for same index, for very high threshold
                if calculate_iou(get_label_box(rel_obj_label), get_label_box(next_rel_obj_label),
                                 is_original=True) > 0.8:
                    is_overlap = True
                    break

            # only add in if no other labels overlaps with current label and share same index
            if not is_overlap:
                concise_rel_obj_data.append(rel_obj_label)

        with open(os.path.join(rel_obj_labels_filepath, label_filename), 'w') as rel_obj_file:
            for label in concise_rel_obj_data:
                for val in label:
                    rel_obj_file.write(f"{val} ")
                rel_obj_file.write("\n")


def is_obj_overlap_pred(obj_label, pred_label, IoU_threshold=0.5):
    """
    Check if the obj label overlaps sufficiently with the pred label
    """
    # check if out of bound
    if get_centre_x(obj_label) < get_x_min(pred_label) or get_centre_x(obj_label) > get_x_max(pred_label) or \
            get_centre_y(obj_label) < get_y_min(pred_label) or get_centre_y(obj_label) > get_y_max(pred_label):
        return False

    # check if sufficient IoU
    # obj must be second input for IoU to be calculated correctly
    # value is amount of obj box overlapped as percentage.
    if calculate_iou(xyxy_cxcywh(*get_label_box(pred_label)),
                     get_label_box(obj_label),
                     is_only_extension=True) < IoU_threshold:
        return False

    return True


def cluster_algo(obj_label_data, pred_label_data):
    """
    Given 2 lists of data, return a dict mapping predicate indices to lists of overlapping object indices.
    Uses O(nlogn) algorithm for interval overlap detection.

    Args:
        obj_label_data: List of object labels in [class, cx, cy, w, h] format
        pred_label_data: List of predicate labels in [class, cx, cy, w, h] format

    Returns:
        dict: Mapping predicate indices to lists of overlapping object indices
    """
    dict_predIndex_to_objIndex_list = {}

    # Convert predicate boxes to xyxy format and add original indices
    xyxy_index_pred_label_data = [
        [label[0], *cxcywh_to_xyxy(*label[1:]), i]
        for i, label in enumerate(pred_label_data)
    ]
    # Sort by x_min
    xyxy_index_pred_label_data.sort(key=lambda x: x[1])  # x[1] is x_min after conversion

    # Add indices to object labels
    index_obj_label_data = [[*label, i] for i, label in enumerate(obj_label_data)]
    # Sort by center_x
    index_obj_label_data.sort(key=lambda x: x[1])  # x[1] is cx

    starting_obj_index = 0

    for cur_sorted_pred_index in range(len(xyxy_index_pred_label_data)):
        cur_xyxy_pred_label = xyxy_index_pred_label_data[cur_sorted_pred_index]
        original_pred_index = cur_xyxy_pred_label[-1]

        # Get predicate box boundaries
        x_min, y_min, x_max, y_max = cur_xyxy_pred_label[1:5]

        # Update starting_obj_index - skip objects that are too far left
        while (starting_obj_index < len(index_obj_label_data) and
               index_obj_label_data[starting_obj_index][1] - index_obj_label_data[starting_obj_index][3] / 2 < x_min):
            starting_obj_index += 1

        # Check objects that might overlap
        cur_obj_index = starting_obj_index
        while cur_obj_index < len(index_obj_label_data):
            cur_obj_label = index_obj_label_data[cur_obj_index]
            obj_cx = cur_obj_label[1]
            obj_cy = cur_obj_label[2]

            # Break if object is too far right
            if obj_cx - cur_obj_label[3] / 2 > x_max:
                break

            # Check y-overlap
            if y_min <= obj_cy <= y_max:
                # Get original boxes for IoU calculation
                pred_box = pred_label_data[original_pred_index][1:]  # cx, cy, w, h
                obj_box = cur_obj_label[1:5]  # cx, cy, w, h

                # Calculate IoU
                iou = calculate_iou(pred_box, obj_box, is_only_extension=True)

                if iou >= 0.5:
                    if original_pred_index not in dict_predIndex_to_objIndex_list:
                        dict_predIndex_to_objIndex_list[original_pred_index] = []
                    dict_predIndex_to_objIndex_list[original_pred_index].append(cur_obj_label[-1])

            cur_obj_index += 1

    return dict_predIndex_to_objIndex_list



def convert_index_to_data(dict_predIndex_to_obj_index, pred_label_data, pred_label_dict,
                          object_label_data, obj_label_dict):
    """
    cannot use dict
    can have multiple same predicate label, but different overlap
    idea: one list for predicate label
    one list for dict of objects
    """

    predicate_list = []
    object_dict_to_boundingbox_list = []

    for predIndex in dict_predIndex_to_obj_index.keys():
        pred_label_index = get_label_index(pred_label_data[predIndex])
        pred_label = pred_label_dict[pred_label_index]
        predicate_list.append(pred_label)

        cur_dict = defaultdict(list)

        for objIndex in dict_predIndex_to_obj_index[predIndex]:
            obj_label_index = get_label_index(object_label_data[objIndex])
            obj_label = obj_label_dict[obj_label_index]
            cur_dict[obj_label].append(get_label_box(object_label_data[objIndex]))
        object_dict_to_boundingbox_list.append(cur_dict)

    return predicate_list, object_dict_to_boundingbox_list


def generate_json_data_from_yolo(yolo_data_path):
    """
    Json file
    { "imgName": [
        {
        "predicate": "predicate1 in img1",
        "object": [{
        "name": "name of the object",
        "x": 100,
        "y": 200,
        "w": 40,
        "h" 50
        },...
        ]
        ]
	}

	each img map to a list of relationships
	each relationships contain 1 predicate key and 1 object key
	object map to a list of dictionary, key is the name, x,y,w,h

    """

    # formatted structure to be inputted into the json
    cur_json_list = defaultdict(list)

    # first generate the proper rel_obj_labels folder to get all the relationships bounding box
    convert_predData_to_relObjData(yolo_data_path)

    image_folder = os.path.join(yolo_data_path, "images")

    obj_label_dict = create_label_dict(os.path.join(yolo_data_path, "obj_labels.txt"))
    pred_label_dict = create_label_dict(os.path.join(yolo_data_path, "pred_labels.txt"))

    for image_filename in tqdm(os.listdir(image_folder), "labelling images"):

        if not image_filename.endswith(('.jpg', '.jpeg', '.png')):
            continue

        # record all relationship in this image
        relationDictList = []

        # Construct the full image path
        img_path = os.path.join(image_folder, image_filename)

        # Read the image
        img = cv2.imread(img_path)

        height, width = img.shape[:2]

        txt_filename = image_filename.split('.')[0] + ".txt"

        obj_label_data = read_labels_from_file(os.path.join(yolo_data_path, "obj_labels", txt_filename),
                                               have_confident=False)
        rel_obj_label_data = read_labels_from_file(os.path.join(yolo_data_path, "rel_obj_labels", txt_filename),
                                                   have_confident=False)

        # get index dict of how relationships overlaps with objects bouding
        index_dict = cluster_algo(obj_label_data, rel_obj_label_data)


        predicate_list, object_dict_to_boundingbox_list = convert_index_to_data(index_dict, rel_obj_label_data,
                                                                                pred_label_dict, obj_label_data,
                                                                                obj_label_dict)

        # each predicate give an dict, keys are object overlap with current predicate
        for cur_pred_label, cur_obj_dict in zip(predicate_list, object_dict_to_boundingbox_list):
            relationDict = {}
            relationDict["predicate"] = cur_pred_label
            relationDict["object"] = []

            for object_label, list_of_boundingbox in cur_obj_dict.items():
                # each object can have a list of bounding box, each is an instance
                cur_obj_label = object_label
                for x, y, w, h in list_of_boundingbox:
                    relationDict["object"].append({
                        "name": cur_obj_label,
                        "x": x * width,
                        "y": y * height,
                        "w": w * width,
                        "h": h * height
                    })
            relationDictList.append(relationDict)
        cur_json_list[image_filename] = relationDictList

    print(relationDictList)

    # Print example of how to read the generated file
    with open("relationships.json", 'w') as f:
        print(json.dump(cur_json_list, f, indent=2))





if __name__ == "__main__":
    # we choose a specific image with groups of bounding box
    img_filename = "000000000632.jpg"
    txt_filename = "000000000632.txt"

    datapath = "D:\Shui Jie\PHD school\Computational Vision\PKU_CV_project\YOLO_SG\sample_data"

    generate_json_data_from_yolo(datapath)

    obj_label_data = read_labels_from_file(os.path.join(datapath, "obj_labels", txt_filename),
                                           have_confident=False)
    rel_obj_label_data = read_labels_from_file(os.path.join(datapath, "rel_obj_labels", txt_filename),
                                               have_confident=False)









    # print(index_dict)

    # obj_label_data = read_labels_from_file(os.path.join(datapath, "obj_labels", txt_filename), have_confident=False)
    # pred_label_data = read_labels_from_file(os.path.join(datapath, "rel_labels", txt_filename), have_confident=False)
    #
    #
    # obj_label_dict = create_label_dict(os.path.join(datapath, "obj_labels.txt"))
    # pred_label_dict = create_label_dict(os.path.join(datapath, "pred_labels.txt"))
    #
    # index_dict = cluster_algo(obj_label_data, pred_label_data)
    #
    # print(f"index_dict: {index_dict}")
    #
    # final_result_dict = convert_index_to_data(index_dict, pred_label_data, pred_label_dict, obj_label_data, obj_label_dict)
    #
    # print(f"final_result_dict: {final_result_dict}")
