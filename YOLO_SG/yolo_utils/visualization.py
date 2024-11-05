import cv2

import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt

import constant

from .label_utils import *


def get_random_color():
    # Generate a random RGB
    return [int(random.random() * 255) for i in range(3)]


def generate_random_colour_scheme_for_labels(label_dict):
    return [get_random_color() for i in range(len(label_dict.keys()))]


def label_img(label_dict, label_data, img, is_data_from_detection=False, color_list=None):
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


def plot_frequency_bar(strings, frequencies, title="Frequency Distribution",
                       xlabel="Frequency", ylabel="Items",
                       figsize=(10, 8), rotation=0):
    """
    Create a horizontal bar graph using matplotlib with random colors for each bar.

    Args:
        strings (list): List of strings for y-axis labels
        frequencies (list): List of corresponding frequencies
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

    # Show the plot
    plt.show()

def data_stat_graph(define_label_dict, abs_data_path):
    """
    given the absolute data path and the corresponding label,
     generate a horizontal bar graph to display distribution of the data.
    """
    strings, frequencies = label_stats(define_label_dict, abs_data_path, have_confident=False)
    strings = [val for i, val in constant.REL_LABEL_DICT.items()]
    plot_frequency_bar(strings, frequencies)


if __name__ == "__main__":
    # Example usage
    data_directory = './coco_dataset/sample_img_data'
    process_images_and_labels(data_directory, constant.REL_LABEL_DICT)
