"""
This where we utilise all the models we trained to generate a single json gfile containing the clusters of relationships.

1. a single folder containing all the yolo weights, together with their individual labels.txt
2. object models and relationships models needs to be separated
3. reformat both obj data and pred data from models into 2 lists of detections, combine their labels into 2 dictionary
4. using cluster algo to directly to get the clusters
5. convert to json format
"""
import concurrent.futures
import os.path

from ultralytics import *
from yolo_utils import *

def load_yolo_model(model_path):
    """
    Loads multiple YOLO models and their corresponding label dictionaries from a directory structure,
    combining all label dictionaries into a unified mapping.

    The function expects a specific directory structure:
    model_path/
    ├── model1/
    │   ├── labels.txt       # Class definitions for model1
    │   └── weight.pt        # Weights for model1
    ├── model2/
    │   ├── labels.txt       # Class definitions for model2
    │   └── weight.pt        # Weights for model2
    └── ...

    Args:
      model_path (str): Path to the root directory containing model subfolders

    Returns:
      tuple: (
          loaded_Models_list: List of loaded YOLO model objects,
          label_mapping_list: List of dictionaries mapping original label indices to combined indices,
          combine_label_dict: Combined dictionary mapping new indices to label names
      )

    Note:
      - Each model subfolder must contain 'weight.pt' and 'labels.txt' files
      - Labels from all models are merged into a single dictionary with new indices
      - The label_mapping_list maintains the relationship between original and new indices
      - The combine_label_dict provides a unified view of all object classes across models
    """

    # store all the loaded models
    loaded_Models_list = []
    # store all the individual label dict
    label_dict_list = []
    # store all the mapping from original label dict index to the combined label dict index
    label_mapping_list = []
    # the final combined label dict
    combine_label_dict ={}

    model_folder_path_list = [os.path.join(model_path, f) for f in os.listdir(model_path)]

    for model_subfolder_path in model_folder_path_list:
        model_weights_path = os.path.join(model_subfolder_path, "weight.pt")
        model_labels_path = os.path.join(model_subfolder_path, "labels.txt")

        yolo_model = YOLO(model_weights_path)
        loaded_Models_list.append(yolo_model)

        cur_label_dict = create_label_dict(model_labels_path)
        label_dict_list.append(cur_label_dict)


    # like a counter, generate the new label index when we combine the label dict
    new_label_index = 0
    for cur_label_dict in label_dict_list:
        map_label_index_to_new_index = {}
        for original_label_index, label in cur_label_dict.items():
            combine_label_dict[new_label_index] = label
            map_label_index_to_new_index[original_label_index] = new_label_index
            new_label_index += 1

        label_mapping_list.append(map_label_index_to_new_index)

    return loaded_Models_list, label_mapping_list, combine_label_dict


def parallel_yolo_detection(loaded_Models_list, label_mapping_list, image):
    """
    Performs parallel object detection using multiple YOLO models and remaps their
    label indices to a unified labeling scheme.

    This function:
    1. Runs detection on the input image using all loaded YOLO models in parallel
    2. Remaps each model's detection labels using its corresponding label mapping
    3. Combines all detections into a single unified output

    Args:
        loaded_Models_list (list): List of loaded YOLO model objects
        label_mapping_list (list): List of dictionaries mapping original model indices
                                 to combined label indices
        image (numpy.ndarray): Input image for detection (BGR format)

    Returns:
        list: List of detections where each detection is a dictionary containing:
            {
                'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
                'confidence': float,        # Detection confidence score
                'class_id': int            # Remapped class ID from combined label dict
            }

    Note:
        - Each model's detections are remapped using its corresponding label_mapping
        - The returned class_ids correspond to the combined label dictionary
        - Bounding box coordinates are in [x1, y1, x2, y2] format
    """

    all_detections = []

    def process_single_model(args):
        model, label_mapping = args
        # Get predictions from the model
        results = model(image, verbose=False, conf=0.2)[0]  # [0] to get first image results

        # Convert predictions to standard format and remap class IDs
        detections = []
        for box in results.boxes:
            # get the normalised values of the box
            x, y, w, h = box.xywhn[0].cpu().numpy()  # Get box coordinates
            confidence = float(box.conf[0])  # Get confidence score
            orig_class_id = int(box.cls[0])  # Get original class ID

            # Remap the class ID using the mapping dictionary
            new_class_id = label_mapping[orig_class_id]

            # a size 6 array
            detections.append([
                new_class_id,
                x, y, w, h,
                confidence
            ])

        return detections

    # Run all models in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create list of (model, mapping) tuples for parallel processing
        model_mapping_pairs = zip(loaded_Models_list, label_mapping_list)

        # Process all models in parallel and gather results
        detection_results = list(executor.map(process_single_model, model_mapping_pairs))

        # Combine all detections into a single list
        for result in detection_results:
            all_detections.extend(result)

    return all_detections


def yolo_sg_application(obj_loaded_Models_list, obj_label_mapping_list, obj_combine_label_dict,
                        rel_loaded_Models_list, rel_label_mapping_list, rel_combine_label_dict,
                        img_path, is_save_label_img=False):
    """
    Performs parallel object and relationship detection using YOLO models and generates scene graph data.

    Uses concurrent threading to run object and relationship detection simultaneously for better performance.
    Optionally saves labeled images showing detected objects and relationships.

    Args:
        obj_loaded_Models_list (list): List of loaded object detection YOLO models
        obj_label_mapping_list (list): Label mappings for object detection models
        obj_combine_label_dict (dict): Combined label dictionary for objects
        rel_loaded_Models_list (list): List of loaded relationship detection YOLO models
        rel_label_mapping_list (list): Label mappings for relationship detection models
        rel_combine_label_dict (dict): Combined label dictionary for relationships
        img (numpy.ndarray): Input image array
        is_save_label_img (bool): Whether to save visualization of detections

    Returns:
        dict: Generated scene graph data in JSON format
    """
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor

    if not os.path.exists("testing_images"):
        create_folder("testing_images")

    # Define tasks for parallel execution
    def object_detection_task():
        obj_labels = parallel_yolo_detection(obj_loaded_Models_list, obj_label_mapping_list, img_path)

        if is_save_label_img:
            out_path = "testing_images/label_img_obj"
            if not os.path.exists(out_path):
                create_folder(out_path)
            print(f"saving object detection visualization to {out_path}")

            # Avoid reading image again since we already have it
            result_img = label_img(obj_combine_label_dict, obj_labels, cv2.imread(img_path),
                                   is_data_from_detection=True)
            cv2.imwrite(os.path.join(out_path, os.path.basename(img_path)), result_img)

        return obj_labels

    def relationship_detection_task():
        rel_labels = parallel_yolo_detection(rel_loaded_Models_list, rel_label_mapping_list, img_path)

        if is_save_label_img:
            out_path = "testing_images/label_img_rel"
            if not os.path.exists(out_path):
                create_folder(out_path)
            print(f"saving relationship detection visualization to {out_path}")

            # Avoid reading image again since we already have it
            result_img = label_img(rel_combine_label_dict, rel_labels, cv2.imread(img_path),
                                   is_data_from_detection=True)
            cv2.imwrite(os.path.join(out_path, os.path.basename(img_path)), result_img)

        return rel_labels

    # Execute both detection tasks in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        obj_future = executor.submit(object_detection_task)
        rel_future = executor.submit(relationship_detection_task)

        try:
            # Wait for both tasks to complete and get results
            obj_label_data = obj_future.result()
            rel_label_data = rel_future.result()
        except Exception as e:
            print(f"Error during parallel detection: {str(e)}")
            raise

    # Generate JSON data using results from both detections
    json_data = direct_generate_json_data_from_yolo(
        img_path,
        obj_label_data,
        rel_label_data,
        obj_combine_label_dict,
        rel_combine_label_dict,
        output_path="testing_images/out_json"
    )

    return json_data


if __name__ == "__main__":
    obj_model_folder_path = "weights/object_models"
    rel_model_folder_path = "weights/relationship_models"
    obj_loaded_Models_list, obj_label_mapping_list, obj_combine_label_dict = load_yolo_model(obj_model_folder_path)
    rel_loaded_Models_list, rel_label_mapping_list, rel_combine_label_dict = load_yolo_model(rel_model_folder_path)

    testing_img_folder = "testing_images/images"

    label_img_folder = "testing_images/labelled images"

    img_filenames = os.listdir(testing_img_folder)

    # Setup progress bar
    pbar = tqdm(img_filenames, desc="Processing Images")

    total_time = 0
    num_images = len(img_filenames)

    for img_filename in pbar:

        # for testing
        # if img_filename != "classroom.png":
        #     continue

        start_time = time.time()
        img_path = os.path.join(testing_img_folder, img_filename)

        yolo_sg_application(obj_loaded_Models_list, obj_label_mapping_list, obj_combine_label_dict,
                            rel_loaded_Models_list, rel_label_mapping_list, rel_combine_label_dict, img_path,
                            is_save_label_img=True)

        # label data
        # obj_label_data = parallel_yolo_detection(loaded_Models_list, label_mapping_list, img_path)
        #
        # result_img = label_img(combine_label_dict, label_data, cv2.imread(img_path), is_data_from_detection=True)
        #
        # cv2.imwrite(os.path.join(label_img_folder, f"labelled_{img_filename}"), result_img)

        # Calculate time for this iteration
        iteration_time = time.time() - start_time
        total_time += iteration_time

        # Update progress bar with current FPS
        current_fps = 1.0 / iteration_time if iteration_time > 0 else 0
        pbar.set_postfix({'Current FPS': f'{current_fps:.2f}'})

    # Calculate final metrics
    average_time = total_time / num_images
    fps = 1.0 / average_time if average_time > 0 else 0

    print(f"\nProcessing Complete!")
    print(f"Average time per image: {average_time:.3f} seconds")
    print(f"Average FPS: {fps:.2f}")

