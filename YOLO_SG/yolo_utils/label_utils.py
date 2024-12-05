import os
import torch

# only print if code is testing
def test_log(log_stmt, is_log_printing):
    if is_log_printing:
        print(log_stmt)

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

# get the value for the label
def get_label_index(label):
    return int(label[0])


# get the x,y,w,h value from the label
def get_label_box(label):
    return label[1:5]

def get_x_min(label):
    return label[1]

def get_y_min(label):
    return label[2]

def get_x_max(label):
    return label[3]

def get_y_max(label):
    return label[4]

def get_centre_x(label):
    return label[1]

def get_centre_y(label):
    return label[2]



def label_stats(label_dict, data_folder_path, have_confident=True):
    """
    Provide a summary of how many labels present in total in the given files.
    :param label_dict: a dictionary mapping index to label
    :param data_folder_path: folder should contain a subfolder named labels
    :param have_confident: detected labels have confident, while ground truth labels dont
    :return: None
    """

    src = "TRAINING"
    if have_confident:
        src = "TESTING"

    label_count = [0 for i in label_dict]
    background = 0
    label_folder = os.path.join(data_folder_path, "labels")

    # get all the labels from both files
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    for label_file in label_files:
        if label_file == "labels.txt":
            # there is a labels.txt file which do not contain labels
            continue
        label_file_path = os.path.join(data_folder_path, "labels", label_file)
        label_list = read_labels_from_file(label_file_path, have_confident)

        if len(label_list) == 0:
            background += 1

        for label in label_list:
            cur_label_index = get_label_index(label)
            label_count[cur_label_index] += 1


    print(f"SUMMARY FOR EACH {src} LABEL")
    for index in range(len(label_count)):
        cur_label = label_dict[index]
        cur_count = label_count[index]
        print(f"there is a total of {cur_count} instance of {cur_label}")
    print(f"there is a total of {background} background images")
    print()

    return label_dict, label_count


def check_valid_label(label_folder_path, label_dict, have_confident=False):
    """
    check if exist label in folder_path that has index out of range.
    """
    label_files = [f for f in os.listdir(label_folder_path) if f.endswith('.txt')]
    for file in label_files:
        filepath = os.path.join(label_folder_path, file)
        label_data_list = read_labels_from_file(filepath, have_confident=have_confident)
        for label_data in label_data_list:
            if get_label_index(label_data) not in label_dict.keys():
                return False

    return True


def get_img_from_label(label_file, image_folder_path, is_log_printing=False):
    img_file_name = os.path.splitext(label_file)[0]
    # Look for the image file in the specified folder
    for file in os.listdir(image_folder_path):
        if file.startswith(img_file_name):
            test_log(f"Found matching image file {file} in '{image_folder_path}'", is_log_printing)
            return os.path.join(image_folder_path, file)
    else:
        # Handle the case when no matching image file is found
        test_log(f"No matching image file found for '{img_file_name}' in '{image_folder_path}'", is_log_printing)
    return ""


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

    xA = min(x1 + w1/2, x2 + w2/2)
    xB = max(x1 - w1/2, x2 - w2/2)

    yA = min(y1 + h1/2, y2 + h2/2)
    yB = max(y1 - h1/2, y2 - h2/2)

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

