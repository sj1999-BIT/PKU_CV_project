SRC_DIR := src
SOURCES := $(SRC_DIR)/*.py
TRAIN_SRC := $(SRC_DIR)/train.py
DATASET_SRC := $(SRC_DIR)/dataset.py
RELS_FILE := relationships.json
GLOVE_FILE := glove.6B.50d.txt
DATASET_FILE := dataset.bin
METRICS_DIR := metrics
VISUALIZE_SRC := $(SRC_DIR)/visualize.py
VISUALIZE_DIR := visuals 
INFERENCE_SRC := $(SRC_DIR)/inference.py
PREDS_TXT := preds.txt

$(DATASET_FILE): $(GLOVE_FILE) $(DATASET_SRC) $(PREDS_TXT)
	python $(DATASET_SRC) --input $(RELS_FILE) --output $(DATASET_FILE) --glove $(GLOVE_FILE) --allowed_preds $(PREDS_TXT)

dataset: $(DATASET_FILE)

train: $(DATASET_FILE) 
	python $(TRAIN_SRC) --input $(DATASET_FILE) --split 10 --learning_rate 0.0001 --layer_1 1000 --layer_2 1000 --layer_3 100 --metrics_dir $(METRICS_DIR) | tee out.log

visualize: $(VISUALIZE_SRC)
	python $(VISUALIZE_SRC) --metrics $(METRICS_DIR) --output $(VISUALIZE_DIR) --dataset $(DATASET_FILE) --confusion_matrix $(METRICS_DIR)/metrics_100.dat

inference: $(INFERENCE_SRC)
	python $(INFERENCE_SRC) --model models/model_9.pth --glove $(GLOVE_FILE) --dataset $(DATASET_FILE) --layer_1 100 --layer_2 100 --layer_3 100 
