SRC_DIR := src
TRAIN_SRC := $(SRC_DIR)/train.py
RELS_FILE := relationships.json
GLOVE_FILE := glove.6B.50d.txt

train: $(TRAIN_SRC) $(RELS_FILE) $(GLOVE_FILE)
	python $(TRAIN_SRC) --input $(RELS_FILE) --glove=$(GLOVE_FILE)
