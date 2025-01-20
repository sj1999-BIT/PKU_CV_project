# YOLO-SG: An Efficient Framework for Scene Graph Generation

## Abstract
Scene graph generation (SGG) is the task of detecting object pairs and their relations in a visual medium, widely used for captioning, generation, and visual question answering. 2D scene graph generation is a subtask that focuses on the generation of a 2D graph given an image.
While development on models capable of performing 2D SGG have improved in both accuracy and speed, the computational complexity of the problem and the inherently long-tailed distribution of large, available datasets has led to generation speed and accuracy less than ideal for real-time use. Mainstream approaches focus on two-stage generation, where object detection is performed first, followed by a series of comparisons for relation inference.
However, these have an inherent drawback where the computational complexity of detecting $n$ objects and their relationships is $n^2$. More recent models have utilized encoder-decoder structures to reduce generation into a 1-stage problem.
Unfortunately, the computation required for these architectures is still to high. Additionally, the model bias caused by long-tailed data distributions remains a key problem in both approaches.
In this work, we propose YOLO-SG, a novel SGG framework capable of operating in real-time by decoupling object detection and relation detection and by performing relationship inference with multiple detection models in parallel. Our proposal seeks to both alleviate the effects of the long-tailed distribution problem and perform high speed inference.
Preliminary experiments on the Visual Genome 1.2 dataset demonstrate that YOLO-SG is able to achieve competitive performance with state-of-the-art models while maintaining high inference speed.

### Prerequisites
- Python 3.12
## Getting Started
1. Run `pip install -r requirements.txt` to install the required packages.

2. Download and combine The Visual Genome 1.2 dataset which can be found [here](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html).
    - Download parts 1 and 2 of the images provided in **Version 1.2** of the dataset
    - Unzip and combine the two parts into a single folder, named `VG_100K` and place it in the YOLO-SG directory
Or just run the following in the YOLO-SG directory:
```bash
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images.zip
unzip images2.zip
cd VG_100K_2
find . -maxdepth 1 -type f -print0 | xargs -0 -I {} mv {} ../VG_100K
cd -
rm -r VG_100K_2
```
3. Due to the large size of certain files such as model weights, instead of being included in the repository, they are stored in a Google Drive folder. The files can be downloaded [here](https://drive.google.com/drive/folders/1CcEHKfdlTWDlrXxJtMrGxfra2AdZZ6oi?usp=sharing). Download the following and **place them in the YOLO-SG** directory:
    - ground_truth.json
    - for_inference.zip (needs to be unzipped)
    - glove.6B.zip (needs to be unzipped)
    - the YOLO-SG/weights directory (needs to be combined with the one in the repository)

4. Run `python3 main.py` in the YOLO-SG directory to run the evaluation performed in the paper for mean Recall@50. The results will be saved in the `evaluation_results.json` file. 
5. To analyze the results, run the `read_evaluation_results.ipynb` notebook.

*Note*: To change the value of $K$, change the `K` variable in the `main.py` file.
