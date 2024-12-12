import os
import cv2
import tqdm
import random

def get_random_color():
    # Generate a random RGB
    return [int(random.random() * 255) for i in range(3)]

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
    visualised_SG_data("./")