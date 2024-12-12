from yolo_sg_pipeline import *
from evaluate_pipeline import *

if __name__ == "__main__":
    
    # imgpath = "./test/000000000139.jpg"
    
    # json_data = yolo_sg_application(imgpath)
    
    # print(json_data)

    testing_img_folder = "./VG_100K" #VG dataset image folder no subdirectories

    img_filenames = os.listdir(testing_img_folder) #list filenames
    print(len(img_filenames)) #debug, should be 100K

    # Setup progress bar
    pbar = tqdm(img_filenames, desc="Processing Images")

    total_time = 0
    num_images = len(img_filenames)

    for img_filename in pbar:

        start_time = time.time()
        img_filepath = os.path.join(testing_img_folder, img_filename)

        json_data = yolo_sg_application(img_filepath) #get yolo clusters (SJ)
        
        #TODO:
        # pass json data to alexay to get triplets
        # compare with ground truth
        # prediction = alexay_get_triplets
        recall, precision = evaluate([prediction], K=20)

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
    
    #TODO:
    # output results




