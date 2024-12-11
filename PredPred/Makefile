SRC_DIR := src
VG_DIR := ~/Projects/vg_dataset
SOURCES := $(SRC_DIR)/*.py
TRAIN_SRC := $(SRC_DIR)/train.py
RELS_FILE := $(VG_DIR)/relationships.json
GLOVE_FILE := $(VG_DIR)/glove.6B.50d.txt
DATA_DIR := ../SplitDataset/out
SUBSET ?= subset_1
DATASET := dataset.bin
METRICS_DIR := metrics
MODEL_DIR := models
VISUALIZE_SRC := $(SRC_DIR)/visualize.py
VISUALIZE_DIR := visuals
INFERENCE_SRC := $(SRC_DIR)/inference.py
DEVICE := cuda
LOG_FILE := out.log

train: $(DATASET_FILE) 
	python $(TRAIN_SRC) --input $(DATA_DIR)/$(SUBSET)/$(DATASET) --split 10 --learning_rate 0.0001 --layer_1 300 --layer_2 100 --layer_3 50 --metrics_dir $(METRICS_DIR)/$(SUBSET) --device $(DEVICE) --epoch_count 1000 --model_dir $(MODEL_DIR)/$(SUBSET) --dropout 0 | tee $(LOG_FILE)

train0: 
	python $(TRAIN_SRC) --input $(DATA_DIR)/subset_0/$(DATASET) --split 10 --learning_rate 0.0001 --layer_1 200 --layer_2 75 --layer_3 30 --metrics_dir $(METRICS_DIR)/subset_0 --device $(DEVICE) --epoch_count 500 --model_dir $(MODEL_DIR)/subset_0 --dropout 0.3 | tee $(LOG_FILE)

train1: 
	python $(TRAIN_SRC) --input $(DATA_DIR)/subset_1/$(DATASET) --split 10 --learning_rate 0.0001 --layer_1 300 --layer_2 100 --layer_3 50 --metrics_dir $(METRICS_DIR)/subset_1 --device $(DEVICE) --epoch_count 1000 --model_dir $(MODEL_DIR)/subset_1 --dropout 0.1 | tee $(LOG_FILE)

train2: 
	python $(TRAIN_SRC) --input $(DATA_DIR)/subset_2/$(DATASET) --split 10 --learning_rate 0.00001 --layer_1 300 --layer_2 100 --layer_3 50 --metrics_dir $(METRICS_DIR)/subset_2 --device $(DEVICE) --epoch_count 1000 --model_dir $(MODEL_DIR)/subset_2 --dropout 0.3 | tee $(LOG_FILE)

train3: 
	python $(TRAIN_SRC) --input $(DATA_DIR)/subset_3/$(DATASET) --split 10 --learning_rate 0.00001 --layer_1 300 --layer_2 100 --layer_3 50 --metrics_dir $(METRICS_DIR)/subset_3 --device $(DEVICE) --epoch_count 1000 --model_dir $(MODEL_DIR)/subset_3 --dropout 0.3 | tee $(LOG_FILE)

train4:
	python $(TRAIN_SRC) --input $(DATA_DIR)/subset_4/$(DATASET) --split 10 --learning_rate 0.00001 --layer_1 150 --layer_2 70 --layer_3 20 --metrics_dir $(METRICS_DIR)/subset_4 --device $(DEVICE) --epoch_count 10000 --model_dir $(MODEL_DIR)/subset_4 --dropout 0.3 | tee $(LOG_FILE)

visualize: $(VISUALIZE_SRC)
	python $(VISUALIZE_SRC) --metrics $(METRICS_DIR)/$(SUBSET) --output $(VISUALIZE_DIR)/$(SUBSET) --dataset $(DATA_DIR)/$(SUBSET)/$(DATASET) --confusion_matrix $(METRICS_DIR)/$(SUBSET)/metrics_00569.bin

inference: $(INFERENCE_SRC)
	python $(INFERENCE_SRC) --model models/model_9.pth --glove $(GLOVE_FILE) --dataset $(DATASET_FILE) --layer_1 100 --layer_2 100 --layer_3 100 

.PHONY: clean
clean:
	rm -rf $(METRICS_DIR)
	rm -rf $(VISUALIZE_DIR)
	rm -rf $(LOG_FILE)
	rm -rf $(MODEL_DIR)

