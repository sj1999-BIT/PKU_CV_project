# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
import os
import random

from PIL import Image
from tqdm import tqdm

import numpy as np
import torchvision.transforms as T

import cv2
import torch

from models import build_model



'''
python inference_with_video.py --img_path SJ_img/split.jpg --resume ckpt/checkpoint0149.pth
'''

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='SJ_img/split.jpg',
                        help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='ckpt/checkpoint0149.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser


transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    lambda x: x.cuda()
])
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
    return b

# VG classes
CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
            'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
            'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
            'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
            'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
            'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
            'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
            'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
            'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
            'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
            'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
            'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
            'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
            'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
            'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
            'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
            'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
            'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
            'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
args = parser.parse_args()
model, _, _ = build_model(args)

# not sure if this can pass the model to cuda
model.cuda()

ckpt = torch.load(args.resume)
model.load_state_dict(ckpt['model'])
model.eval()

"""
For generating a random color
"""
def generate_random_rgb():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

"""
Calculate if current box overlaps sufficiently, return bool
"""
def is_overlapped(box1, box2, threshold=0.5):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area

    return (intersection_area / union_area) > threshold


"""
function draws bounded box with label on frame
"""
def draw_box_with_label(frame, label, color, sxmin, symin, sxmax, symax):
    cv2.rectangle(frame, (sxmin, symin), (sxmax, symax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(frame, (sxmin, symin - label_size[1] - 10), (sxmin + label_size[0], symin), (0, 255, 0))
    cv2.putText(frame, label, (sxmin, symin - 5), font, font_scale, color, font_thickness)
    return frame

def draw_line_with_label(frame, color, start_point, end_point, label):
    # Draw the line
    cv2.line(frame, start_point, end_point, color=color, thickness=2)
    # Calculate the mid-point for the label
    mid_point = (int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2))
    # Put the label at the mid-point
    cv2.putText(frame, label, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame



def detectTR(frame):
    """
    Function returns a list of triplets
    triplets in a form of numpy
    """


    # first convert image from np array to PIL image
    # Convert BGR to RGB
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(frame)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

    # keep only predictions with 0.+ confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))
    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], pil_image.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], pil_image.size)

    topk = 10
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(
        -probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[
            0])[:topk]
    keep_queries = keep_queries[indices]


    # based on the function, we need
    # 1. keep_queries: provides idx for us to locate the obj_labels for sub, obj, pred
    # 2. sub_bboxes_scaled[indices] and obj_bboxes_scaled[indices]: contains x1, y1, x2, y2 for subject bounding boxes
    # 3. probas_sub, probas_obj, probas: contains list of indexes, can be used to locate the highest probable label.
    # 4. frame: the numpy image frame

    return keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices], probas_sub, probas_obj, probas

def label_frame(frame, keep_queries, sub_bboxes_list, obj_bboxes_list, prob_sub, prob_obj, prob_pred):
    # use same colour for label
    DICT_LABEL_TO_COLOR = {}
    # use IoU to check if current box overlaps
    DICT_LABEL_TO_BOX = {}

    for idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
            zip(keep_queries, sub_bboxes_list, obj_bboxes_list):

        sxmin = int(sxmin.item())
        symin = int(symin.item())
        sxmax = int(sxmax.item())
        symax = int(symax.item())

        oxmin = int(oxmin.item())
        oymin = int(oymin.item())
        oxmax = int(oxmax.item())
        oymax = int(oymax.item())

        subject = CLASSES[prob_sub[idx].argmax()]
        object = CLASSES[prob_obj[idx].argmax()]
        predicate = REL_CLASSES[prob_pred[idx].argmax()]

        # default random color
        color = generate_random_rgb()
        if subject in DICT_LABEL_TO_COLOR.keys():
            color = DICT_LABEL_TO_COLOR[subject]

        if object in DICT_LABEL_TO_COLOR.keys():
            color = DICT_LABEL_TO_COLOR[object]

        if subject not in DICT_LABEL_TO_BOX.keys() or not is_overlapped((sxmin, symin, sxmax, symax), DICT_LABEL_TO_BOX[subject]):
            frame = draw_box_with_label(frame, subject, color, sxmin, symin, sxmax, symax)
            DICT_LABEL_TO_BOX[subject] = (sxmin, symin, sxmax, symax)
        if object not in DICT_LABEL_TO_BOX.keys() or not is_overlapped((oxmin, oymin, oxmax, oymax), DICT_LABEL_TO_BOX[object]):
            frame = draw_box_with_label(frame, object, color, oxmin, oymin, oxmax, oymax)
            DICT_LABEL_TO_BOX[object] = (oxmin, oymin, oxmax, oymax)

        # store color
        DICT_LABEL_TO_COLOR[subject] = color
        DICT_LABEL_TO_COLOR[object] = color

        # Calculate the center of the boxes
        subject_center = (int((sxmin + sxmax) / 2), int((symin + symax) / 2))
        object_center = (int((oxmin + oxmax) / 2), int((oymin + oymax) / 2))

        # Draw line with label
        frame = draw_line_with_label(frame, color, subject_center, object_center, predicate)

    return frame


def get_video_properties(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # If properties are 0, try to read the first frame to get dimensions
    if width == 0 or height == 0 or fps == 0:
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            # Reset video capture to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            raise ValueError("Unable to read video frame")

    # If FPS is still 0, set a default value
    if fps == 0:
        fps = 30
        print("Warning: Could not detect FPS. Using default value of 30.")

    return width, height, fps


def extract_frame(input_video_path, frame_number, output_image_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Set the video position to the specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    if ret:
        # Save the frame as a PNG file
        cv2.imwrite(output_image_path, frame)
        print(f"Frame {frame_number} saved as {output_image_path}")
    else:
        print(f"Error: Could not read frame {frame_number}")

    # Release the video capture object
    cap.release()


"""
Functions finds mp4 file in input path
runs model over each frame to label all triplets
render a new video and place in output path
"""
def process_video(input_path, output_directory):
    # Extract filename from input path
    input_filename = os.path.basename(input_path)

    # Create output path
    output_path = os.path.join(output_directory, input_filename)

    # Open the input video
    cap = cv2.VideoCapture(input_path)

    print(f"check video opened: {cap.isOpened()}")

    # Get video properties
    try:
        width, height, fps = get_video_properties(cap)
    except ValueError as e:
        print(f"Error: {str(e)}")
        cap.release()
        return

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process video with progress bar
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            # Process the frame to get the detected output
            keep_queries, sub_bboxes_list, obj_bboxes_list, prob_sub, prob_obj, prob_pred = detectTR(frame)

            # label the frame for output
            processed_frame = label_frame(frame, keep_queries, sub_bboxes_list, obj_bboxes_list, prob_sub, prob_obj, prob_pred)

            # Display the frame
            cv2.imshow('Video Processing', processed_frame)

            # Write the frame
            # out.write(processed_frame)

            pbar.update(1)


    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    input_abs_dir = "D:\Shui Jie\PHD school\Computational Vision\PKU_CV_project\RELTR\SJ_video\input"
    output_abs_dir = "D:\Shui Jie\PHD school\Computational Vision\PKU_CV_project\RELTR\SJ_video\output"

    video_filename = "women_dog.mp4"

    input_abs_path = os.path.join(input_abs_dir, video_filename)

    # extract_frame(input_abs_path, 63, os.path.join(output_abs_dir, f"frame_{63}.png"))

    process_video(input_abs_path, output_abs_dir)

    # image_filepath = "D:\Shui Jie\PHD school\Computational Vision\PKU_CV_project\RELTR\SJ_video\output/frame_63.png"
    #
    # img = Image.open(image_filepath)
    #
    # cv2.imshow('Image', detectTR(np.array(img)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
