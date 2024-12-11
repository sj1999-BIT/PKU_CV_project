from yolo_sg_pipeline import *

if __name__ == "__main__":


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
        img_filepath = os.path.join(testing_img_folder, img_filename)

        yolo_sg_application(img_filepath, is_save_label_img=True)

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




