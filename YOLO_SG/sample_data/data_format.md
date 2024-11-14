Copy# Dataset Structure Documentation

## Overview
This dataset is organized to support visual relationship detection, containing images and their corresponding object and relationship annotations in YOLO-compatible format.


## Directory Structure
```
dataset 
   ├── images/               # All image files
   ├── obj_labels/          # YOLO format object annotations
   ├── rel_labels/          # Relationship triplet annotations
   ├── obj_labels.txt       # Object class definitions
   └── pred_labels.txt      # Predicate class definitions
```


## File Descriptions

### Label Definition Files
- **obj_labels.txt**: Contains object class labels, one per line
```
person
car
chair
...
```
- **pred_labels.txt**: Contains predicate class labels, one per line
```
on
holding
near
...
```

### Annotation Files

#### Object Annotations (obj_labels/)
- Files named corresponding to images (e.g., `image1.jpg` → `image1.txt`)
- YOLO format: `<label_index> <x_center> <y_center> <width> <height>`
- Each line represents one object
- Example:
```
0 0.5 0.5 0.3 0.4
1 0.7 0.8 0.2 0.3
```
#### Relationship Annotations (rel_labels/)
- Files named corresponding to images (e.g., `image1.jpg` → `image1.txt`)
- Format: `<obj_index> <subj_index> <pred_label_index>`
- Each line represents one relationship triplet
- Example: 
```
0 1 2
1 3 0
```
#### Example
- images/img1.jpg (image file)
- obj_labels.txt (content)
  - ```
    person
    car
    chair
    ```
- pred_labels.txt(content)
  - ```
    on
    holding
    near
    ```
- obj_labels/img1.txt(content)
  - ```
    0 0.5 0.5 0.3 0.4
    1 0.7 0.8 0.2 0.7
    2 0.5 0.5 0.3 0.4
    2 0.7 0.8 0.2 0.3
    ```
- pred_labels.txt(content)
  - ```
    0 1 2
    0 3 1
    ```
- interpretation
  - in this image contains 4 objects (0: person, 1: car, 2: chair)
  - 2 triplets:
    - `0 1 2`: person (line 0), car (line 1), near (pred_index 0)
    - `0 3 2`: person (line 0), chair (line 3), holding (pred_index 1)
  - conclusion: 
    - image contain 2 relationship:
      - person (box: 0.5 0.5 0.3 0.4) near car (box: 0.7 0.8 0.2 0.7)
      - person (box: 0.5 0.5 0.3 0.4) holding chair (box: 0.7 0.8 0.2 0.3)


## Notes
- Object and subject indices in relationship files reference line numbers (0-based) in the corresponding object annotation file
- Label indices reference line numbers in:
- `obj_labels.txt` for object classes
- `pred_labels.txt` for predicate classes
- All indices are 0-based