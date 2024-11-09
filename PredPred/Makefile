SRC_DIR := src
SOURCES := $(SRC_DIR)/*.py
TRAIN_SRC := $(SRC_DIR)/train.py
DATASET_SRC := $(SRC_DIR)/dataset.py
RELS_FILE := relationships.json
GLOVE_FILE := glove.6B.50d.txt
DATASET_FILE := dataset.bin

$(DATASET_FILE): $(GLOVE_FILE)
	python $(DATASET_SRC) --input $(RELS_FILE) --output $(DATASET_FILE) --glove $(GLOVE_FILE)

dataset: $(DATASET_FILE)

train: $(DATASET_FILE) 
	python $(TRAIN_SRC) --input $(DATASET_FILE) --layer_1 10000 --layer_2 10000 --split 10 | tee output.log
