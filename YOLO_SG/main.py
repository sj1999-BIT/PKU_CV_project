from yolo_sg_pipeline import *
from evaluate_pipeline import *
import PredPred.src.inference as head
import random

if __name__ == "__main__":
    
    # imgpath = "./test/000000000139.jpg"
    
    # json_data = yolo_sg_application(imgpath)
    
    # print(json_data)

    testing_img_folder = "./VG_100K" #VG dataset image folder no subdirectories

    img_filenames = os.listdir(testing_img_folder) #list filenames
    # print('no. of images: ', len(img_filenames)) #debug, should be 100K

    # Setup progress bar
    pbar = tqdm(img_filenames, desc="Processing Images")


    total_time = 0
    num_images = len(img_filenames)

    args = type('obj', (object, ), {
        "models": "../PredPred/models.json",
        "device": "cuda",
        # EDIT THESE:
        "glove": "./glove.6B/glove.6B.50d.txt",
        "datasets": "./for_inference/data",
        "weights": "./for_inference/models",
    })
    runner = head.Runner(args)
    
    recall_scores = []
    precision_scores = []
    i = 0 #debug
    for img_filename in pbar:
        # if i < 100:
        #     i+=1
        # else:
        #     break
        #randomly select 10%
        i = random.randint(1, 10)
        if i > 3:
            continue

        start_time = time.time()
        img_filepath = os.path.join(testing_img_folder, img_filename)
        try:
            json_data = yolo_sg_application(img_filepath) #get yolo clusters (SJ)
        except Exception as e:
            continue
        predicate = json_data
        # print() #debug
        # print('json_data: ', json_data) #debug
        prediction = runner.run_single_image(img_filename, json_data[img_filename])
        # print('prediction: ', prediction) #debug
        
        K = 50
        recall, precision = evaluate(prediction, K)
        # print(f'Recall@{K}: {recall:.2f}\nPrecision@{K}: {precision:.2f}\n')
        recall_scores.append(recall)
        precision_scores.append(precision)
        # for testing
        # if img_filename != "classroom.png":
        #     continue

        # Calculate time for this iteration
        iteration_time = time.time() - start_time
        total_time += iteration_time

        # Update progress bar with current FPS
        current_fps = 1.0 / iteration_time if iteration_time > 0 else 0
        pbar.set_postfix({'Current FPS': f'{current_fps:.2f}'})
        
        # break #debug

    # Calculate final metrics
    average_time = total_time / num_images
    fps = 1.0 / average_time if average_time > 0 else 0

    print(f"\nProcessing Complete!")
    print(f"Average time per image: {average_time:.3f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    #TODO:
    # output results
    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_precision = sum(precision_scores) / len(precision_scores)
    max_recall = max([score for score in recall_scores if score is not 1])
    # print(f'mean Recall@{K}: {mean_recall:.2f}\nmean Precision@{K} {mean_precision:.2f}')
    
    results = {
      "K": K,
      "recall_scores": recall_scores,
      "precision_scores": precision_scores,
      "mean_recall": mean_recall,
      "mean_precision": mean_precision,
      "max_recall": max_recall
    }
    
    try:
      with open("evaluation_results.json", "r+") as f:
          data = json.load(f)
          data["experiments"].append(results)
          f.seek(0)
          json.dump(data, f, indent=None)
          f.truncate()
    except FileNotFoundError:
      with open("evaluation_results.json", "w") as f:
          json.dump({"experiments": [results]}, f, indent=None)
    




