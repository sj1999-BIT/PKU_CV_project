# evaluation/benchmarking section of the pipeline
import matplotlib.pyplot as plt
import json

class SynonimSet:
    def __init__(self):
        self.word_to_class = {}
        self.classes = []
        self.unused = []
        self.locked = False

    def load(self, file):
        with open(file, "r") as file:
            for line in file:
                x = str.split(line, ": ")
                repr = str.strip(x[0])
                syns = []
                if len(x) > 1:
                    for s in str.split(x[1], ", "):
                        syns.append(str.strip(s))
                # print(repr, ":", syns)        
                self.add_class(repr, syns)
            self.locked = True

    def class_count(self):
        return len(self.classes) - len(self.unused)

    def representatives(self):
        res = []
        for c in self.classes:
            if len(c) > 0:
                res.append(c[0])
        return res

    def union(self, word1, word2):
        c1 = self.get_class_id(word1)
        c2 = self.get_class_id(word2)
        if c1 == c2 or word1 == word2:
            return
        for w in self.classes[c2]:
            self.word_to_class[w] = (c1, len(self.classes[c1]))
            self.classes[c1].append(w)
        self.classes[c2] = []
        self.unused.append(c2)

    def add_class(self, representative, synonims=[]):
        if self.locked:
            return

        self.add_word(representative)
        id = self.get_class_id(representative)
        for s in synonims:
            if s in self.word_to_class:
                self.union(representative, s)
            else:
                self.word_to_class[s] = (id, len(self.classes[id]))
                self.classes[id].append(s)

    def add_word(self, word):
        if word in self.word_to_class:
            return
        id = 0
        if len(self.unused) > 0:
            id = self.unused.pop(0)
        else:
            id = len(self.classes)
            self.classes.append([])

        self.classes[id].append(word)
        self.word_to_class[word] = (id, 0)
        return

    def save(self, dir, name):
        with open(f"{dir}/{name}", "w") as f:
            for words in self.classes:
                if len(words) == 1:
                    f.write(f"{words[0]}\n")
                else:
                    f.write(f"{words[0]}: ")
                    for i, w in enumerate(words[1:]):
                        f.write(f"{w}")
                        if i + 1 != len(words[1:]):
                            f.write(", ")
                        else:
                            f.write("\n")

    def remove_class(self, representative):
        id = self.word_to_class[representative][0]
        self.classes[id] = []
        self.unused.append(id)
        del self.word_to_class[representative]

    def get_repr(self, word):
        if word not in self.word_to_class:
            return word
        x =  self.classes[self.word_to_class[word][0]][0]
        return x
        
    def get_repr_by_id(self, class_id):
        return self.classes[class_id][0]

    def get_synonims(self, word):
        return self.classes[self.word_to_class[word][0]][1:]

    def get_class_id(self, word):
        return self.word_to_class[word][0]

    def get_word_id(self, word):
        return self.word_to_class[word][1]

predsym = SynonimSet()
predsym.load("../SplitDataset/preds.txt")
objsym = SynonimSet()
objsym.load("../SplitDataset/objs.txt")


to_omit = None
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
    global to_omit
    if not to_omit:
        with open('../to_omit.txt', 'r') as omit_file:
            words = omit_file.readlines()
            to_omit = [word.strip() for word in words]

    for filename, pred_list in predictions.items():
        if filename not in ground_truth_dict:
            continue
        
        gt_relationships = ground_truth_dict[filename][0]['relationships']
        gt_triplets = {(rel['subject']['name'], rel['predicate'], rel['object']['name']) for rel in gt_relationships}
        # print("==============GT_RAW", gt_triplets)
        
        gt_triplets = {(
            objsym.get_repr(rel['subject']['name']), 
            predsym.get_repr(rel['predicate']), 
            objsym.get_repr(rel['object']['name'])
        ) for rel in gt_relationships}
        # print("==============GT_SYN", gt_triplets)
        gt_triplets = {triplet for triplet in gt_triplets if not any(omit_word in triplet for omit_word in to_omit)}
        
        sorted_preds = sorted(pred_list, key=lambda x: x['confidence'], reverse=True)[:K]
        pred_triplets = {(pred['subject']['name'], pred['predicate'], pred['object']['name']) for pred in sorted_preds}
        # print("==============GT_FILTER", gt_triplets)
        # print("==============PRED", pred_triplets)

        matches = gt_triplets & pred_triplets
        total_correct += len(matches)
        total_preds += len(gt_triplets)
        # print('matches found: ', matches) #debug

    recall = 1 if total_preds == 0 else total_correct / total_preds 
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
    global to_omit
    if not to_omit:
        with open('./to_omit.txt', 'r') as omit_file:
            words = omit_file.readlines()
            to_omit = [word.strip() for word in words]
    
    for filename, pred_list in predictions.items():
        if filename not in ground_truth_dict:
            continue

        gt_relationships = ground_truth_dict[filename][0]['relationships']
        gt_triplets = {(rel['subject']['name'], rel['predicate'], rel['object']['name']) for rel in gt_relationships}
        gt_triplets = {triplet for triplet in gt_triplets if not any(omit_word in triplet for omit_word in to_omit)}

        sorted_preds = sorted(pred_list, key=lambda x: x['confidence'], reverse=True)[:K]
        pred_triplets = {(pred['subject']['name'], pred['predicate'], pred['object']['name']) for pred in sorted_preds}
        

        matches = gt_triplets & pred_triplets
        total_correct += len(matches)
        total_preds += len(gt_triplets)

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

ground_truth = None 
#the key function in this file
def evaluate(predictions, K):
    global ground_truth

    if ground_truth is None:
        with open('../ground_truth.json', 'r') as gt_file:
            ground_truth = json.load(gt_file)

    recall = recall_at_k(predictions, ground_truth, K)
    precision = precision_at_k(predictions, ground_truth, K)
    # f1 = f1_score(recall, precision)
    return recall, precision
