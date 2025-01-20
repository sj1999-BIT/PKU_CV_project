# YOLO-SG: An Efficient Framework for Scene Graph Generation

## Abstract
Scene graph generation (SGG) is the task of detecting object pairs and their relations in a visual medium, widely used for captioning, generation, and visual question answering. 2D scene graph generation is a subtask that focuses on the generation of a 2D graph given an image.
While development on models capable of performing 2D SGG have improved in both accuracy and speed, the computational complexity of the problem and the inherently long-tailed distribution of large, available datasets has led to generation speed and accuracy less than ideal for real-time use. Mainstream approaches focus on two-stage generation, where object detection is performed first, followed by a series of comparisons for relation inference.
However, these have an inherent drawback where the computational complexity of detecting $n$ objects and their relationships is $n^2$. More recent models have utilized encoder-decoder structures to reduce generation into a 1-stage problem.
Unfortunately, the computation required for these architectures is still to high. Additionally, the model bias caused by long-tailed data distributions remains a key problem in both approaches.
In this work, we propose YOLO-SG, a novel SGG framework capable of operating in real-time by decoupling object detection and relation detection and by performing relationship inference with multiple detection models in parallel. Our proposal seeks to both alleviate the effects of the long-tailed distribution problem and perform high speed inference.
Preliminary experiments on the Visual Genome 1.2 dataset demonstrate that YOLO-SG is able to achieve competitive performance with state-of-the-art models while maintaining high inference speed.

## Getting Started
### Prerequisites
- Py
#### Dataset
The Visual Genome 1.2 dataset can be found [here](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html).
- Download parts 1 and 2 of the images provided in **Version 1.2** of the dataset
- Unzip and combine the two parts into a single folder, named `VG_100K`
 

Due to the large size of certain files, instead of being included in the repository, they are stored in a Google Drive folder. The files can be downloaded from the following [link](https://drive.google.com/drive/folders/1CcEHKfdlTWDlrXxJtMrGxfra2AdZZ6oi?usp=sharing).

