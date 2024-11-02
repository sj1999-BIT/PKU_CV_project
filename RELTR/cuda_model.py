import os
import cv2
import torch
import argparse

import numpy as np
import torchvision.transforms as T

from PIL import Image

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
CLASSES = ['N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
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

cur_parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
args = cur_parser.parse_args()
model, _, _ = build_model(args)

# push to cuda
model.cuda()

ckpt = torch.load(args.resume)
model.load_state_dict(ckpt['model'])
model.eval()


def model_inference(input_data):
    """
    Perform model inference on different types of input
    Args:
        input_data: Can be PIL Image, numpy array, or filepath (str)
    Returns:
        model outputs
    """
    # Convert input to PIL Image based on type
    if isinstance(input_data, str):
        # Handle filepath
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"File not found: {input_data}")
        pil_image = Image.open(input_data).convert('RGB')

    elif isinstance(input_data, np.ndarray):
        # Handle numpy array
        # Check if image is BGR (from cv2) and convert to RGB if needed
        if len(input_data.shape) == 3 and input_data.shape[2] == 3:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(input_data)

    elif isinstance(input_data, Image.Image):
        # Handle PIL Image
        pil_image = input_data

    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}. "
                        "Must be PIL Image, numpy array, or filepath.")

    # Ensure image is in RGB mode
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # mean-std normalize the input image (batch-size: 1)
    img = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

    return outputs
