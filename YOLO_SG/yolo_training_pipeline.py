from ultralytics import YOLO

from yolo_utils import *


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def yolo_training_pipeline(define_label_dict, original_data_path, epochs=300,
                           img_size=640, initial_model_weight='yolo11n.pt', output_format="onnx"):
    """
    function to train a yolo based on a simple data folder
    :param define_label_dict: a dict mapping label index to labels.
                              E.g. {0: 'tiles', 1: 'cracks', 2: 'empty', 3: 'chipped_off'}
    :param original_data_path: data path to folder containing original data arranged as such.
                                            └── images: folder contain all images for training
                                            └── labels: ground truth labels, not all images need a corresponding label.
    :param epochs: determine number of epoch for training (optional).
    :param img_size: determine input size nXn of images. All Images will be transformed to this size (optional).
    :param initial_model_weight: string for initialised weight of model (optional).
    :param output_format: format of output weight (optional).
    :return: path to dir where the best trained weights are stored.
    """
    # Generate the new folder name
    new_folder_name = f'train_{os.path.basename(original_data_path)}'

    # Combine the new folder name with the parent directory to get the full path
    trainable_data_path = os.path.join(os.path.dirname(original_data_path), new_folder_name)

    if not check_valid_label(original_data_path, define_label_dict):
        # exist invalid label
        print("stop training due to invalid label")
        return

    # prepare the data for model training, get the yaml file path as its needed for training
    yaml_file_path = full_yolo_data_prep_pipeline(define_label_dict, original_data_path, trainable_data_path)

    model = YOLO(initial_model_weight)  # load a pretrained model (recommended for training)

    # Train the model
    # normally training is done within 300epoch
    results = model.train(data=yaml_file_path, epochs=epochs, imgsz=img_size, batch=8, cache=False)
    # export the model to ONNX format
    trained_weight_path = model.export(format=output_format)

    return trained_weight_path

if __name__ == '__main__':
    # given the folder containing the original data: images and labels 2 subfolders
    original_datapath = "./coco_dataset/balanced_balanced_yolo_5k_data"

    # model_weight_path = "trained_usable_weights/normal_training_pavement_trial8_300epoch.onnx"

    # define the label dictionary for this dataset
    # define_label_dict = {0: 'tiles', 1: 'cracks', 2: 'empty', 3: 'chipped_off'}

    # # export the model to ONNX format
    # trained_yolo_weights_filepath = yolo_training_pipeline(define_label_dict, original_datapath, img_size=640)

    # trained_yolo_weights_filepath = yolo_training_pipeline(constant.REL_LABEL_DICT, original_datapath, img_size=640)

    # print(f'path is {trained_yolo_weights_filepath}')

    # define_original_data_path = "./coco_dataset/train_yolo_5k_data/train"
    #
    #
    #
    # # define the label dictionary for this dataset
    # # define_label_dict = {0: 'crack', 1: 'chipped_off', 2: 'net_crack'}
    #
    # data_stat_graph(constant.REL_LABEL_DICT, original_datapath)

    _, label_count = label_stats(constant.REL_LABEL_DICT, original_datapath, have_confident=False)
    new_path = create_balance_data(original_datapath, label_count, new_data_path=None, min_val=100)
    data_stat_graph(constant.REL_LABEL_DICT, new_path)



