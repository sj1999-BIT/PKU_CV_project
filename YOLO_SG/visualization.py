import os
import cv2
import random
import constant


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


if __name__ == "__main__":
    # Example usage
    data_directory = './coco_dataset/sample_img_data'
    process_images_and_labels(data_directory, constant.OBJ_CLASS_DICT)
