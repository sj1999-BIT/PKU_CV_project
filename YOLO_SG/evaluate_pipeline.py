# evaluation/benchmarking section of the pipeline
import matplotlib.pyplot as plt

def recall_at_k(predictions, ground_truth_dict, K) -> float:
    """
    Calculate Recall@K for scene graph evaluation.
    
    Args:
        predictions (dict): Predictions for images, where keys are filenames and values are lists of triplets.
        ground_truth_dict (dict): Ground truth relationships indexed by filename.
        K (int): The top-K predictions to consider.
        
    Returns:
        float: The Recall@K value.  When more than one image is passed in, it returns the mean Recall@K value.
    """
    total_correct = 0
    total_preds = 0

    for filename, pred_list in predictions.items():
        if filename not in ground_truth_dict:
            continue
        
        gt_relationships = ground_truth_dict[filename][0]['relationships']
        gt_triplets = {(rel['subject']['name'], rel['predicate'], rel['object']['name']) for rel in gt_relationships}
        
        sorted_preds = sorted(pred_list, key=lambda x: x['confidence'], reverse=True)[:K]
        pred_triplets = {(pred['subject']['name'], pred['predicate'], pred['object']['name']) for pred in sorted_preds}

        total_correct += len(gt_triplets & pred_triplets)
        total_preds += len(gt_triplets)

    recall = 0.0 if total_preds == 0 else total_correct / total_preds 
    return recall


def precision_at_k(predictions, ground_truth_dict, K) -> float:
    """
    Calculate Precision@K for scene graph evaluation.
    
    Args:
        predictions (dict): Predictions for images, where keys are filenames and values are lists of triplets.
        ground_truth_dict (dict): Ground truth relationships indexed by filename.
        K (int): The top-K predictions to consider.
        
    Returns:
        float: The Precision@K value.  When more than one image is passed in, it returns the mean Precision@K value.
    """
    total_correct = 0
    total_preds = 0

    for filename, pred_list in predictions.items():
        if filename not in ground_truth_dict:
            continue
        
        gt_relationships = ground_truth_dict[filename][0]['relationships']
        gt_triplets = {(rel['subject']['name'], rel['predicate'], rel['object']['name']) for rel in gt_relationships}

        sorted_preds = sorted(pred_list, key=lambda x: x['confidence'], reverse=True)[:K]
        pred_triplets = {(pred['subject']['name'], pred['predicate'], pred['object']['name']) for pred in sorted_preds}

        total_correct += len(gt_triplets & pred_triplets)
        total_preds += len(pred_triplets)

    precision = 0.0 if total_preds == 0 else total_correct / total_preds
    return precision


def f1_score(recall, precision):
    if recall + precision == 0:
        return 0.0 
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

#visualisation
def plot_recall_precision(k_values, recall_scores, precision_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, recall_scores, label="Recall@K", marker='o')
    plt.plot(k_values, precision_scores, label="Precision@K", marker='s')

    plt.title("Recall@K and Precision@K")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()
    plt.show()

#the key function in this file
def evaluate(predictions, ground_truths, K):
    recall = recall_at_k(predictions, ground_truths, K)
    precision = precision_at_k(predictions, ground_truths, K)
    # f1 = f1_score(recall, precision)
    return recall, precision
