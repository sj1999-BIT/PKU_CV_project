import json
import math
import os
import time
import sys
import shutil

import cv2
import torch
import yaml
import random

from collections import defaultdict

from tqdm import tqdm


# only print if code is testing
def test_log(log_stmt, is_log_printing):
    if is_log_printing:
        print(log_stmt)


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


def create_folder(folder_path, is_log_printing=False):
    # Create the output folder if it doesn't exist
    if not os.path.exists(folder_path):
        test_log(f"new folder created {folder_path}", is_log_printing)
        os.makedirs(folder_path)


def read_labels_from_file(file_path, have_confident=True):
    labels = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if have_confident:
                    class_id, x, y, w, h, confid = map(float, parts)
                else:
                    class_id, x, y, w, h = map(float, parts)
                labels.append((int(class_id), x, y, w, h))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return labels


def create_label_dict(file_path):
    label_dict = {}
    with open(file_path, 'r') as f:
        classes = f.read().strip().split()
        for idx, class_name in enumerate(classes):
            label_dict[idx] = class_name
    return label_dict


# get the x,y,w,h value from the label
def get_label_box(label):
    return label[1:5]


# get the value for the label
def get_label_index(label):
    return int(label[0])


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


def cxcywh_to_xyxy(x_center, y_center, box_width, box_height):
    """
    Convert bounding box from center format (x_center, y_center, box_height, box_width)
    to corner format (x1, y1, x2, y2).

    Parameters:
    - x_center: The x-coordinate of the box center.
    - y_center: The y-coordinate of the box center.
    - box_height: The height of the bounding box.
    - box_width: The width of the bounding box.

    Returns:
    - x1: The x-coordinate of the top-left corner of the box.
    - y1: The y-coordinate of the top-left corner of the box.
    - x2: The x-coordinate of the bottom-right corner of the box.
    - y2: The y-coordinate of the bottom-right corner of the box.
    """
    x1 = (x_center - box_width / 2)
    y1 = (y_center - box_height / 2)
    x2 = (x_center + box_width / 2)
    y2 = (y_center + box_height / 2)

    return x1, y1, x2, y2


def xyxy_cxcywh(x1, y1, x2, y2):
    """
    Convert bounding box from corner format (x1, y1, x2, y2) to center format
    (x_center, y_center, box_height, box_width).

    Parameters:
    - x1: The x-coordinate of the top-left corner of the box.
    - y1: The y-coordinate of the top-left corner of the box.
    - x2: The x-coordinate of the bottom-right corner of the box.
    - y2: The y-coordinate of the bottom-right corner of the box.

    Returns:
    - x_center: The x-coordinate of the box center.
    - y_center: The y-coordinate of the box center.
    - box_height: The height of the bounding box.
    - box_width: The width of the bounding box.
    """
    x_center = x1 + (x2 - x1) / 2
    y_center = y1 + (y2 - y1) / 2
    box_height = y2 - y1
    box_width = x2 - x1

    return x_center, y_center, box_height, box_width


def merge_cxcywh(box1, box2):
    """
    Generate a larger bounding box that covers both input boxes.

    Parameters:
    - box1: A tuple (x_center1, y_center1, box_height1, box_width1) representing the first bounding box.
    - box2: A tuple (x_center2, y_center2, box_height2, box_width2) representing the second bounding box.

    Returns:
    - A tuple (x_center, y_center, box_height, box_width) of the new bounding box that covers both.
    """

    # Convert inputs to tensors
    box1 = torch.tensor(box1)
    box2 = torch.tensor(box2)

    # Unpack the input boxes
    x_center1, y_center1, box_width1, box_height1 = box1
    x_center2, y_center2, box_width2, box_height2 = box2

    # Calculate the corners of the first box and second box
    x1_1, y1_1, x2_1, y2_1 = cxcywh_to_xyxy(x_center1, y_center1, box_width1, box_height1)
    x1_2, y1_2, x2_2, y2_2 = cxcywh_to_xyxy(x_center2, y_center2, box_width2, box_height2)

    # Find the minimum and maximum corners to create the bounding box
    # bounded to 0 and 1
    x1_min = torch.clamp(torch.min(torch.stack((x1_1, x1_2))), 0, 1)
    y1_min = torch.clamp(torch.min(torch.stack((y1_1, y1_2))), 0, 1)
    x2_max = torch.clamp(torch.max(torch.stack((x2_1, x2_2))), 0, 1)
    y2_max = torch.clamp(torch.max(torch.stack((y2_1, y2_2))), 0, 1)

    # Calculate the center and size of the new bounding box
    x_center, y_center, box_width, box_height = xyxy_cxcywh(x1_min, y1_min, x2_max, y2_max)

    return x_center, y_center, box_height, box_width


def calculate_iou(box1, box2, is_only_extension=False, is_original=False):
    # Calculate the intersection over union (IOU) of two bounding boxes
    # print(box1)
    # print(box2)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xA = min(x1 + w1 / 2, x2 + w2 / 2)
    xB = max(x1 - w1 / 2, x2 - w2 / 2)

    yA = min(y1 + h1 / 2, y2 + h2 / 2)
    yB = max(y1 - h1 / 2, y2 - h2 / 2)

    inter_area = (xB - xA) * (yB - yA)

    box1_area = w1 * h1
    box2_area = w2 * h2

    if is_original:
        return inter_area / (float(box2_area) + float(box1_area) - inter_area)

    if is_only_extension:
        return inter_area / float(box2_area)

    one_overlap = 0
    if float(box1_area) != 0:
        one_overlap = inter_area / float(box1_area)

    two_overlap = 0
    if float(box2_area) != 0:
        two_overlap = inter_area / float(box2_area)

    return max(one_overlap, two_overlap)


# def cluster_algo(obj_label_data, pred_label_data):
#     """
#     Given 2 lists of data, return a dict mapping predicate indices to lists of overlapping object indices.
#     Uses O(nlogn) algorithm for interval overlap detection.
#
#     Args:
#         obj_label_data: List of object labels in [class, cx, cy, w, h] format
#         pred_label_data: List of predicate labels in [class, cx, cy, w, h] format
#
#     Returns:
#         dict: Mapping predicate indices to lists of overlapping object indices
#     """
#     dict_predIndex_to_objIndex_list = {}
#
#     # Convert predicate boxes to xyxy format and add original indices
#     xyxy_index_pred_label_data = [
#         [label[0], *cxcywh_to_xyxy(*label[1:5]), i]
#         for i, label in enumerate(pred_label_data)
#     ]
#     # Sort by x_min
#     xyxy_index_pred_label_data.sort(key=lambda x: x[1])  # x[1] is x_min after conversion
#
#     # Add indices to object labels
#     index_obj_label_data = [[*label, i] for i, label in enumerate(obj_label_data)]
#     # Sort by center_x
#     index_obj_label_data.sort(key=lambda x: x[1])  # x[1] is cx
#
#     starting_obj_index = 0
#
#     for cur_sorted_pred_index in range(len(xyxy_index_pred_label_data)):
#         cur_xyxy_pred_label = xyxy_index_pred_label_data[cur_sorted_pred_index]
#         original_pred_index = cur_xyxy_pred_label[-1]
#
#         # Get predicate box boundaries
#         x_min, y_min, x_max, y_max = cur_xyxy_pred_label[1:5]
#
#         # Update starting_obj_index - skip objects that are too far left
#         while (starting_obj_index < len(index_obj_label_data) and
#                index_obj_label_data[starting_obj_index][1] - index_obj_label_data[starting_obj_index][3] / 2 < x_min):
#             starting_obj_index += 1
#
#         # Check objects that might overlap
#         cur_obj_index = starting_obj_index
#         while cur_obj_index < len(index_obj_label_data):
#             cur_obj_label = index_obj_label_data[cur_obj_index]
#             obj_cx = cur_obj_label[1]
#             obj_cy = cur_obj_label[2]
#
#             # Break if object is too far right
#             if obj_cx - cur_obj_label[3] / 2 > x_max:
#                 break
#
#             # Check y-overlap
#             if y_min <= obj_cy <= y_max:
#                 # Get original boxes for IoU calculation
#                 pred_box = pred_label_data[original_pred_index][1:5]  # cx, cy, w, h
#                 obj_box = cur_obj_label[1:5]  # cx, cy, w, h
#
#                 # Calculate IoU
#                 iou = calculate_iou(pred_box, obj_box, is_only_extension=True)
#
#                 if iou >= 0.5:
#                     if original_pred_index not in dict_predIndex_to_objIndex_list:
#                         dict_predIndex_to_objIndex_list[original_pred_index] = []
#                     dict_predIndex_to_objIndex_list[original_pred_index].append(cur_obj_label[-1])
#
#             cur_obj_index += 1
#
#     return dict_predIndex_to_objIndex_list

def calculate_box_boundaries(box, format='cxcywh'):
    """Helper function to get box boundaries in consistent format"""
    if format == 'cxcywh':
        cx, cy, w, h = box
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2
    else:  # xyxy format
        x_min, y_min, x_max, y_max = box
    return x_min, y_min, x_max, y_max


def boxes_overlap(box1, box2, format1='cxcywh', format2='cxcywh'):
    """Check if two boxes overlap in both x and y dimensions"""
    x_min1, y_min1, x_max1, y_max1 = calculate_box_boundaries(box1, format1)
    x_min2, y_min2, x_max2, y_max2 = calculate_box_boundaries(box2, format2)

    x_overlap = (x_min1 <= x_max2) and (x_max1 >= x_min2)
    y_overlap = (y_min1 <= y_max2) and (y_max1 >= y_min2)

    return x_overlap and y_overlap


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
        [label[0], *cxcywh_to_xyxy(*label[1:5]), i]
        for i, label in enumerate(pred_label_data)
    ]
    # Sort by x_min
    xyxy_index_pred_label_data.sort(key=lambda x: x[1])  # x[1] is x_min after conversion

    # Process each predicate
    for pred_idx, pred_data in enumerate(xyxy_index_pred_label_data):
        original_pred_index = pred_data[-1]
        pred_box = pred_label_data[original_pred_index][1:5]  # Original box in cxcywh format

        # Check overlap with all objects
        # (removing the optimization temporarily to ensure we catch all overlaps)
        for obj_idx, obj_label in enumerate(obj_label_data):
            obj_box = obj_label[1:5]  # cx, cy, w, h

            # Check if boxes overlap
            if boxes_overlap(pred_box, obj_box):
                # Calculate IoU if needed
                iou = calculate_iou(pred_box, obj_box, is_only_extension=True)

                if iou >= 0.5:
                    if original_pred_index not in dict_predIndex_to_objIndex_list:
                        dict_predIndex_to_objIndex_list[original_pred_index] = []
                    dict_predIndex_to_objIndex_list[original_pred_index].append(obj_idx)

    return dict_predIndex_to_objIndex_list


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


def generate_json_data_from_yolo(yolo_data_path="./"):
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


def direct_generate_json_data_from_yolo(img_path, obj_label_data, rel_obj_label_data, obj_label_dict, pred_label_dict,
                                        output_path="", save_json=False):
    """
    directly provide the values for generation
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

    # record all relationship in this image
    relationDictList = []

    # Read the image
    img = cv2.imread(img_path)

    height, width = img.shape[:2]

    # get index dict of how relationships overlaps with objects bounding
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
                # Force Python float after multiplication
                # print(F"x {x}, y {y}, w {w}, h {h}, width {width}, height {height}, ")
                relationDict["object"].append({
                    "name": cur_obj_label,
                    "x": float(x * width),
                    "y": float(y * height),
                    "w": float(w * width),
                    "h": float(h * height)
                })
        relationDictList.append(relationDict)

    cur_json_list[os.path.basename(img_path)] = relationDictList

    # print(cur_json_list)

    # print(relationDictList)

    img_filename = os.path.basename(img_path).split(".")[0]

    if save_json:

        if not os.path.exists(output_path):
            create_folder(output_path)

        # Print example of how to read the generated file
        with open(os.path.join(output_path, f"{img_filename}_relationships.json"), 'w') as f:
            json.dump(cur_json_list, f, indent=2)

    return cur_json_list


if __name__ == "__main__":
    # we choose a specific image with groups of bounding box
    generate_json_data_from_yolo()
