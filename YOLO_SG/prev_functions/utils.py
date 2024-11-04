import os
import random
import shutil
import yaml
import glob


def get_labels_file_name(img_filename):
    # Get the base name of the image file (without extension)
    image_name = os.path.splitext(img_filename)[0]

    # Create the text file name by adding ".txt" to the image name
    text_file_name = image_name + ".txt"

    return text_file_name

# only print if code is testing
def test_log(log_stmt, is_log_printing):
    if is_log_printing:
        print(log_stmt)


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

# function generate true based on input percentage
def random_true(true_chance=10):
    # Generate a random number between 0 and 1
    random_number = random.random()

    # Check if the random number is less than 0.1 (10% probability)
    if random_number < true_chance / 100:
        return True
    else:
        return False


def list_cur_dir(is_log_printing=False):
    # Get the current working directory
    current_directory = os.getcwd()

    # List all items (files and directories) in the current directory
    all_items = os.listdir(current_directory)

    # Filter only the directories
    folders = [item for item in all_items if os.path.isdir(os.path.join(current_directory, item))]

    # Print the list of folders
    test_log("Folders in the current directory:", is_log_printing)
    for folder in folders:
        print(folder)


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


def remove_folder(folder_path):
    try:
        # Use shutil.rmtree to remove the folder and its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' removed successfully.")
    except OSError as e:
        print(f"Error: {e}")


def create_folder(folder_path, is_log_printing=False):
    # Create the output folder if it doesn't exist
    if not os.path.exists(folder_path):
        test_log(f"new folder created {folder_path}", is_log_printing)
        os.makedirs(folder_path)


# folder requires all yolo data should have an arrangement of
#   └── images
#   └── obj_labels
def prepare_data_folder(folder_path, is_log_printing=False):
    create_folder(folder_path)
    create_folder(os.path.join(folder_path, "images"))
    create_folder(os.path.join(folder_path, "obj_labels"))


# folder contains the weights and their important information
#   └── label_dictionary
#   └── non_defect_label_index
#   └── weights
def prepare_combined_model_folder(folder_path, is_log_printing=False):
    create_folder(folder_path)
    create_folder(os.path.join(folder_path, "label_dictionary"))
    create_folder(os.path.join(folder_path, "weights"))


def prepare_yolo_data_folder(folder_path, is_log_printing=False):
    """
    folder requires all yolo data should have an arrangement of
    └── train
    └── val
    └── test
        └── images
        └── obj_labels
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


def create_file_list(txt_file_path, target_dir, is_log_printing=False):
    with open(txt_file_path, 'w') as f:
        for file in os.listdir(target_dir):
            input_path = os.path.join(target_dir, file) + "\n"
            test_log(f'path writter: {input_path}', is_log_printing)
            f.write(input_path)


def extract_pavement_images(cam_index, root_dir, target_dir="data", target_folder="extracted", is_log_printing=False):
    count = 0  # File count

    dest_dir = os.path.join(target_dir, target_folder)

    os.makedirs(dest_dir, exist_ok=True)  # Create Destination directory

    for filename in glob.glob(os.path.join(root_dir, f'**/Cam{cam_index}*'),
                              recursive=True):  # Find file recursively in all subfolders (** means all subfolders when recursive=True)
        count += 1
        shutil.copy(filename, dest_dir)  # Copy to destination directory
        test_log("Copied: {}".format(filename), is_log_printing=is_log_printing)

    test_log("Total files copied: {}\n".format(count), is_log_printing=is_log_printing)


def prep_txt_file_for_yolo(target_data_dir, is_log_printing=False):
    """
    all yolo train data should have an arrangement of
    └── datasets
        └── train
        └── val
        └── test
            └── images
            └── obj_labels
    can just perform txt file creation given the main data dir
    :param target_data_dir:
    :return:
    """
    train_dir = os.path.join(target_data_dir, "train")
    val_dir = os.path.join(target_data_dir, "val")
    test_dir = os.path.join(target_data_dir, "test")

    create_file_list(os.path.join(target_data_dir, "train_lighthaus.txt"), os.path.join(train_dir, "images"), is_log_printing)
    create_file_list(os.path.join(target_data_dir, "val_lighthaus.txt"), os.path.join(val_dir, "images"), is_log_printing)
    create_file_list(os.path.join(target_data_dir, "test_lighthaus.txt"), os.path.join(test_dir, "images"), is_log_printing)


def transfer_yolo_data(img_file, folder_path, out_folder_path, is_log_printing=False):
    label_file = os.path.splitext(img_file)[0] + ".txt"

    img_folder = os.path.join(folder_path, "images")
    label_folder = os.path.join(folder_path, "obj_labels")

    # original data path
    cur_img_path = os.path.join(img_folder, img_file)
    cur_label_path = os.path.join(label_folder, label_file)

    # new data path
    new_img_path = os.path.join(out_folder_path, "images", img_file)
    new_label_path = os.path.join(out_folder_path, "obj_labels", label_file)

    # copy data over
    shutil.copy(cur_img_path, new_img_path)
    if not os.path.exists(cur_label_path):
        # no original obj_labels, then just create a empty file
        with open(new_label_path, 'w') as f:
            pass
    else:
        # copy over the label txt file
        shutil.copy(cur_label_path, new_label_path)


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


def separate_data(folder_path, output_folder_path, val_percent=10.0, test_percent=10.0, is_log_printing=False):
    """
    Given a folder with images and obj_labels, copy and separate them into train, val, test
    :param folder_path: folder containing images and label
    """
    clear_folder(output_folder_path)
    prepare_yolo_data_folder(output_folder_path)

    img_folder = os.path.join(folder_path, "images")
    label_folder = os.path.join(folder_path, "obj_labels")

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
        for img_file in img_files:
            if random_true(int(val_percent)):
                transfer_yolo_data(img_file, folder_path, val_out_path)
                val_count += 1
                is_val_empty = False
            elif random_true(int(test_percent)):
                transfer_yolo_data(img_file, folder_path, test_out_path)
                test_count += 1
                is_test_empty = False

    # transfer the rest to train folder
    for img_file in img_files:
        if img_file in os.listdir(val_out_path) or img_file in os.listdir(test_out_path):
            continue
        out_folder_path = os.path.join(output_folder_path, "train")
        transfer_yolo_data(img_file, folder_path, out_folder_path)

    test_log(f'total data separated: train has {len(img_files)-val_count-test_count}, val has {val_count}, test has {test_count}', is_log_printing)


def create_yaml_file(data_path, label_dict, yaml_filename="lighthaus_data.yaml", is_log_printing=False):
    config = {
        'path': data_path,
        'train': "train_lighthaus.txt",
        'val': "val_lighthaus.txt",
        'test': "test_lighthaus.txt",
        'names': {i: label for i, label in enumerate(label_dict)}
    }

    yaml_file_path = os.path.join(data_path, yaml_filename)
    with open(yaml_file_path, 'w') as file:
        test_log(f"yaml created with {config} at {yaml_file_path}", is_log_printing)
        yaml.dump(config, file)

    return yaml_file_path


def read_label_dict_defect_data(label_dict_filepath):
    """
    Given the path to a file containing txt files of each model, generate a dict for label and a list of non_defects
    :param label_dict_filepath: path to file
    :return:label_dict, non_defect_list, inputsize, mapIndexToDefectIndex
    """
    with open(label_dict_filepath, 'r') as f:
        label_dict = {}
        index_To_Defect_Dict = {}
        non_defect_list = []
        label_index = 0
        input_size = -1
        for line in f:
            if (input_size < 0):
                input_size = int(line)
                continue
            # Split the line into X and Y based on space
            label, is_defect = line.split()

            # next value maps the label to defects
            defect_index = int(is_defect)
            is_defect = int(is_defect) == 0

            label_dict[label_index] = label
            if not is_defect:
                non_defect_list.append(label_index)
                index_To_Defect_Dict[label_index] = defect_index
            label_index += 1
        return label_dict, non_defect_list, input_size, index_To_Defect_Dict


def prep_combined_output_folder(output_dir, model_name_list, is_log_printing=False):
    create_folder(output_dir, is_log_printing=is_log_printing)
    for model_name in model_name_list:
        create_folder(os.path.join(output_dir, f"prediction_{model_name}"), is_log_printing=is_log_printing)

def copy_random_images(src_folder, dst_folder, fraction=0.5):
    # Get all files in src_folder
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    # Calculate number of files to copy
    num_files_to_copy = int(len(files) * fraction)

    # Randomly select a subset of files
    files_to_copy = random.sample(files, num_files_to_copy)

    # Create dst_folder if it doesn't exist
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Copy files
    for file in files_to_copy:
        shutil.copy(os.path.join(src_folder, file), dst_folder)

def remove_unlabeled_images(data_folder):
    images_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "obj_labels")
    img_files = [f for f in os.listdir(images_folder)]
    label_files = [f for f in os.listdir(labels_folder)]

    for img in img_files:
        if get_labels_file_name(img) not in label_files:
            os.remove(os.path.join(images_folder, img))



if __name__ == '__main__':
    src_folder = "D:/Shui Jie/career/lighthaus/data/tile_modified_data/Tile_Extracted_Images"
    remove_unlabeled_images(src_folder)
    # def_combined_model_folder_path = "trained_usable_weights/Combined_model_folder"
    # for label_dict_data_file in os.listdir(os.path.join(def_combined_model_folder_path, "label_dictionary")):
    #     read_label_dict_defect_data(os.path.join(def_combined_model_folder_path, "label_dictionary", label_dict_data_file))