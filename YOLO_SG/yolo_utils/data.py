import os
import shutil

import yaml
import random

from tqdm import tqdm


# only print if code is testing
def test_log(log_stmt, is_log_printing):
    if is_log_printing:
        print(log_stmt)

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

    test_log(f'total data separated: train has {len(img_files)-val_count-test_count}, val has {val_count}, test has {test_count}', is_log_printing)

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

    create_file_list(os.path.join(target_data_dir, "train_lighthaus.txt"), os.path.join("train", "images"), is_log_printing)
    create_file_list(os.path.join(target_data_dir, "val_lighthaus.txt"), os.path.join("val", "images"), is_log_printing)
    create_file_list(os.path.join(target_data_dir, "test_lighthaus.txt"), os.path.join("test", "images"), is_log_printing)

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

