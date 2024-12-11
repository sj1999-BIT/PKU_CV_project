import os
import sys
import random
import cv2
import tqdm

import seaborn as sns
import numpy as np
from PIL import Image

rootpath = os.path.join(os.getcwd(), '..')
sys.path.append(rootpath)

from matplotlib import pyplot as plt
import constant
from yolo_utils.label_utils import *


def get_random_color():
    # Generate a random RGB
    return [int(random.random() * 255) for i in range(3)]


def generate_random_colour_scheme_for_labels(label_dict):
    return [get_random_color() for i in range(len(label_dict.keys()))]


def label_img(label_dict, label_data, img, is_data_from_detection=False, color_list=None):
    """
    Draws bounding boxes and labels on an image using YOLO format annotations or detection results.

    This function visualizes object locations by:
    1. Converting YOLO format coordinates (normalized x,y,w,h) to pixel coordinates
    2. Drawing bounding boxes with class labels
    3. Adding confidence scores for detection results

    Args:
        label_dict (dict): Dictionary mapping class indices to label names
        label_data (list): List of labels/detections in YOLO format where each item is either:
            - [label_index, x_center, y_center, width, height] for annotations
            - [label_index, x_center, y_center, width, height, confidence] for detections
        img (numpy.ndarray): Input image to draw on (BGR format)
        is_data_from_detection (bool): If True, expects detection format with confidence scores
        color_list (list, optional): List of BGR colors for each class. If None, generates random colors

    Returns:
        numpy.ndarray: Image with drawn bounding boxes and labels

    Note:
        - Input coordinates should be normalized (0-1) in YOLO format
        - Converts normalized YOLO format (x_center, y_center, width, height) to pixel coordinates
        - For detections, adds "Detected_" prefix and confidence score to labels
        - Uses OpenCV for drawing operations
        - Default text settings: HERSHEY_SIMPLEX font, size 0.5, thickness 2
    """
    img_height, img_width, _ = img.shape

    # generate random colour if not colour specified
    if color_list is None:
        color_list = generate_random_colour_scheme_for_labels(label_dict)

    for label in label_data:
        # if its detected label, added in detection and confidence
        if is_data_from_detection:
            label_index, x_center, y_center, box_width, box_height, confidence = label
            label_text = f"Detected_{label_dict[label_index]} ({confidence})"
        else:
            label_index, x_center, y_center, box_width, box_height = label
            label_text = f"{label_dict[label_index]}"

        label_index = int(label_index)

        # Convert YOLO format to OpenCV format
        x1 = int((x_center - box_width / 2) * img_width)
        y1 = int((y_center - box_height / 2) * img_height)
        x2 = int((x_center + box_width / 2) * img_width)
        y2 = int((y_center + box_height / 2) * img_height)

        # Draw bounding box with specific color
        color = color_list[label_index]
        thickness = 2
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return img


def process_images_and_labels(data_dir, label_dict):
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "obj_labels")
    # Create a directory for labelled images
    labelled_img_dir = os.path.join(data_dir, 'obj_labelled_img')
    os.makedirs(labelled_img_dir, exist_ok=True)

    # Get all image files
    for image_filename in os.listdir(image_dir):
        if image_filename.endswith(('.jpg', '.jpeg', '.png')):
            # Construct the full image path
            img_path = os.path.join(image_dir, image_filename)

            # Read the image
            img = cv2.imread(img_path)

            # Construct the corresponding label file path
            label_file_path = os.path.join(label_dir, os.path.splitext(image_filename)[0] + '.txt')

            if os.path.exists(label_file_path):
                # Read the label data
                label_data = []
                with open(label_file_path, 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        label_data.append(
                            parts)  # Assuming each line has the format: index, x_center, y_center, box_width, box_height, confidence

                # Label the image
                labeled_img = label_img(label_dict, label_data, img)

                # Save the labeled image
                cv2.imwrite(os.path.join(labelled_img_dir, image_filename), labeled_img)

def create_combined_labelled_img(image_filepath, label_dict, original_color_list, detected_color_list,
                                 original_label_data, detected_label_data, output_path):
    # image_name = os.path.basename(image_filepath)
    #
    # output_path = os.path.join(output_folder, image_name)

    if os.path.exists(image_filepath):
        # Load the image
        img = cv2.imread(image_filepath)

        img = label_img(label_dict, original_label_data, img, color_list=original_color_list)
        img = label_img(label_dict, detected_label_data, img, is_data_from_detection=True,
                        color_list=detected_color_list)

        # Save the image with bounding boxes
        cv2.imwrite(output_path, img)
        print(f'image saved at {output_path}')
    print("finished")

def plot_confusion_matrix(tp=0, fp=0, tn=0, fn=0, title='Confusion Matrix', save_dir=None):
    """
    Plot the confusion matrix given results.

    :param tp: True Positive
    :param fp: False Positive
    :param tn: True Negative
    :param fn: False Negative
    :param title: Plot title
    :param save_dir: Directory to save the plot (if None, the plot will be displayed but not saved)
    """
    # Create confusion matrix
    conf_matrix = np.array([[fp, tn],
                            [tp, fn]])
    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)

    # Save or show the plot
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'confusion_matrix_plot.png')
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()


def plot_frequency_bar(strings, frequencies, output_path="frequency_plot.jpg", title="Frequency Distribution",
                       xlabel="Frequency", ylabel="Items",
                       figsize=(10, 8), rotation=0):
    """
    Create and save a horizontal bar graph using matplotlib with random colors for each bar.
    Args:
        strings (list): List of strings for y-axis labels
        frequencies (list): List of corresponding frequencies
        output_path (str): Path where to save the JPG file
        title (str): Title of the graph
        xlabel (str): Label for x-axis (frequency)
        ylabel (str): Label for y-axis (items)
        figsize (tuple): Figure size (width, height)
        rotation (int): Rotation angle for y-axis labels
    """
    # Convert inputs to numpy arrays and ensure proper types
    y = np.arange(len(strings))  # Create numerical y-coordinates
    frequencies = np.array(frequencies, dtype=float)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Generate random colors
    colors = [f'#{np.random.randint(0, 16777215):06x}' for _ in strings]

    # Create horizontal bars with numerical y-coordinates
    bars = ax.barh(y, frequencies, color=colors)

    # Set the y-tick labels to your strings
    ax.set_yticks(y)
    ax.set_yticklabels(strings)

    # Customize the graph
    ax.set_title(title, pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add value labels on end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.02,
                bar.get_y() + bar.get_height() / 2.,
                f'{int(width):,}',
                ha='left',
                va='center')

    # Add some padding to the right for the labels
    plt.margins(x=0.1)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot as JPG with high DPI for better quality
    plt.savefig(output_path, format='jpg', dpi=300, bbox_inches='tight')

    # Close the figure to free memory
    plt.close(fig)


def data_stat_graph(define_label_dict, abs_data_path, output_path="frequency_plot.jpg"):
    """
    given the absolute data path and the corresponding label,
     generate a horizontal bar graph to display distribution of the data.
    """
    strings, frequencies = label_stats(define_label_dict, abs_data_path, have_confident=False)
    strings = [val for i, val in constant.REL_LABEL_DICT.items()]
    plot_frequency_bar(strings, frequencies, output_path=output_path)


def draw_box_with_label(frame, label, color, sxmin, symin, sxmax, symax):
    cv2.rectangle(frame, (sxmin, symin), (sxmax, symax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(frame, (sxmin, symin - label_size[1] - 10), (sxmin + label_size[0], symin), (0, 255, 0))
    cv2.putText(frame, label, (sxmin, symin - 5), font, font_scale, color, font_thickness)
    return frame


def draw_line_with_label(frame, color, start_point, end_point, label):
    # Draw the line
    cv2.line(frame, start_point, end_point, color=color, thickness=2)
    # Calculate the mid-point for the label
    mid_point = (int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2))
    # Put the label at the mid-point
    cv2.putText(frame, label, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def visualised_SG_data(sg_formatted_data_filepath, limit=100):
    """
    given a formatted data dir:
       ├── images/               # All image files
       ├── obj_labels/          # YOLO format object annotations
       ├── pred_labels/          # Relationship triplet annotations
       ├── obj_labels.txt       # Object class definitions
       └── pred_labels.txt      # Predicate class definitions
    generate a new subfolder in the data dir called labelled_images, contains images with the labelled SGG

    can use limit to reduce the number of images generated
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


    # generate new folder to store labelled images
    labelled_image_folder = os.path.join(sg_formatted_data_filepath, "labelled_images")
    if not os.path.exists(labelled_image_folder):
        os.makedirs(labelled_image_folder)

    OBJ_CLASS_DICT = create_label_dict(obj_labels_filepath)
    PRED_CLASS_DICT = create_label_dict(pred_labels_filepath)

    # use same colour for label
    DICT_LABEL_TO_COLOR = {}
    # use IoU to check if current box overlaps
    DICT_LABEL_TO_BOX = {}

    count = 0

    for image_filename in tqdm.tqdm(os.listdir(image_folder), "labelling images"):
        if image_filename.endswith(('.jpg', '.jpeg', '.png')):
            count +=1
            # Construct the full image path
            img_path = os.path.join(image_folder, image_filename)

            # Read the image
            img = cv2.imread(img_path)

            height, width = img.shape[:2]

            label_filename =image_filename.split('.')[0] + ".txt"

            # draw boxes for object
            obj_label_data = read_labels_from_file(os.path.join(obj_labels_folder, label_filename), have_confident=False)

            obj_label_data = [[label_index, cx*width, cy*height, w*width, h*height]
                              for label_index, cx, cy, w, h in obj_label_data]

            for cur_label in obj_label_data:
                label_index, cx, cy, w, h = cur_label
                sxmin, symin, sxmax, symax = cxcywh_to_xyxy(cx, cy, w, h)
                # to image scale
                img = draw_box_with_label(img, OBJ_CLASS_DICT[label_index], get_random_color(),
                                          int(sxmin), int(symin), int(sxmax), int(symax))

            # now get the relationship
            pred_labels = []
            try:
                with open(os.path.join(pred_labels_folder, label_filename), 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        obj, sub, pred = map(int, parts)
                        pred_labels.append((obj, sub, pred))
            except FileNotFoundError:
                print(f"File not found: {os.path.join(pred_labels_folder, label_filename)}")

            # draw lines to attach each box to indicate relationship
            for obj_index, sub_index, pred_index in pred_labels:
                # Calculate the center of the boxes
                subject_center = [int(x) for x in obj_label_data[sub_index][1:3]]
                object_center = [int(x) for x in obj_label_data[obj_index][1:3]]
                predicate = PRED_CLASS_DICT[pred_index]

                # Draw line with label
                img = draw_line_with_label(img, get_random_color(), subject_center, object_center, predicate)

            # save image
            cv2.imwrite(os.path.join(labelled_image_folder, image_filename), img)

            # stop generating images if needed.
            if count > limit:
                break

if __name__ == "__main__":
    # Example usage
    # data_directory = './coco_dataset/sample_img_data'
    # process_images_and_labels(data_directory, constant.REL_LABEL_DICT)
    data_directory = '../sample_data'
    visualised_SG_data(data_directory)
