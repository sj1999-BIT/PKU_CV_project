DATASET := out
SCRIPT := split.py
VG_DIR := ~/Projects/vg_dataset
RELATIONSHIPS := $(VG_DIR)/relationships.json
GLOVE_FILE := $(VG_DIR)/glove.6B.50d.txt
PREDS_TXT := preds.txt
OBJS_TXT:= objs.txt
IMAGE_DATA := $(VG_DIR)/image_data.json
IMAGES := $(VG_DIR)/images
EXTRA_FLAGS ?= 

$(DATASET): $(SCRIPT) $(PREDS_TXT) $(OBJS_TXT)
	python $(SCRIPT) --relationships $(RELATIONSHIPS) --vg $(IMAGES) --objs $(OBJS_TXT) --preds $(PREDS_TXT) --image_data $(IMAGE_DATA) --glove $(GLOVE_FILE) --output $(DATASET) --split --trim --yolo $(EXTRA_FLAGS)

build: $(DATASET)

.PHONY: clean
clean:
	rm -rf $(DATASET)
