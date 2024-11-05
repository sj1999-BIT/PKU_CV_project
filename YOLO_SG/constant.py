REL_LABEL_DICT = {
    0: 'above',
    1: 'across',
    2: 'against',
    3: 'along',
    4: 'and',
    5: 'at',
    6: 'attached to',
    7: 'behind',
    8: 'belonging to',
    9: 'between',
    10: 'carrying',
    11: 'covered in',
    12: 'covering',
    13: 'eating',
    14: 'flying in',
    15: 'for',
    16: 'from',
    17: 'growing on',
    18: 'hanging from',
    19: 'has',
    20: 'holding',
    21: 'in',
    22: 'in front of',
    23: 'laying on',
    24: 'looking at',
    25: 'lying on',
    26: 'made of',
    27: 'mounted on',
    28: 'near',
    29: 'of',
    30: 'on',
    31: 'on back of',
    32: 'over',
    33: 'painted on',
    34: 'parked on',
    35: 'part of',
    36: 'playing',
    37: 'riding',
    38: 'says',
    39: 'sitting on',
    40: 'standing on',
    41: 'to',
    42: 'under',
    43: 'using',
    44: 'walking in',
    45: 'walking on',
    46: 'watching',
    47: 'wearing',
    48: 'wears',
    49: 'with',
}


OBJ_CLASS_DICT= {
    0: 'airplane',
    1: 'animal',
    2: 'arm',
    3: 'bag',
    4: 'banana',
    5: 'basket',
    6: 'beach',
    7: 'bear',
    8: 'bed',
    9: 'bench',
    10: 'bike',
    11: 'bird',
    12: 'board',
    13: 'boat',
    14: 'book',
    15: 'boot',
    16: 'bottle',
    17: 'bowl',
    18: 'box',
    19: 'boy',
    20: 'branch',
    21: 'building',
    22: 'bus',
    23: 'cabinet',
    24: 'cap',
    25: 'car',
    26: 'cat',
    27: 'chair',
    28: 'child',
    29: 'clock',
    30: 'coat',
    31: 'counter',
    32: 'cow',
    33: 'cup',
    34: 'curtain',
    35: 'desk',
    36: 'dog',
    37: 'door',
    38: 'drawer',
    39: 'ear',
    40: 'elephant',
    41: 'engine',
    42: 'eye',
    43: 'face',
    44: 'fence',
    45: 'finger',
    46: 'flag',
    47: 'flower',
    48: 'food',
    49: 'fork',
    50: 'fruit',
    51: 'giraffe',
    52: 'girl',
    53: 'glass',
    54: 'glove',
    55: 'guy',
    56: 'hair',
    57: 'hand',
    58: 'handle',
    59: 'hat',
    60: 'head',
    61: 'helmet',
    62: 'hill',
    63: 'horse',
    64: 'house',
    65: 'jacket',
    66: 'jean',
    67: 'kid',
    68: 'kite',
    69: 'lady',
    70: 'lamp',
    71: 'laptop',
    72: 'leaf',
    73: 'leg',
    74: 'letter',
    75: 'light',
    76: 'logo',
    77: 'man',
    78: 'men',
    79: 'motorcycle',
    80: 'mountain',
    81: 'mouth',
    82: 'neck',
    83: 'nose',
    84: 'number',
    85: 'orange',
    86: 'pant',
    87: 'paper',
    88: 'paw',
    89: 'people',
    90: 'person',
    91: 'phone',
    92: 'pillow',
    93: 'pizza',
    94: 'plane',
    95: 'plant',
    96: 'plate',
    97: 'player',
    98: 'pole',
    99: 'post',
    100: 'pot',
    101: 'racket',
    102: 'railing',
    103: 'rock',
    104: 'roof',
    105: 'room',
    106: 'screen',
    107: 'seat',
    108: 'sheep',
    109: 'shelf',
    110: 'shirt',
    111: 'shoe',
    112: 'short',
    113: 'sidewalk',
    114: 'sign',
    115: 'sink',
    116: 'skateboard',
    117: 'ski',
    118: 'skier',
    119: 'sneaker',
    120: 'snow',
    121: 'sock',
    122: 'stand',
    123: 'street',
    124: 'surfboard',
    125: 'table',
    126: 'tail',
    127: 'tie',
    128: 'tile',
    129: 'tire',
    130: 'toilet',
    131: 'towel',
    132: 'tower',
    133: 'track',
    134: 'train',
    135: 'tree',
    136: 'truck',
    137: 'trunk',
    138: 'umbrella',
    139: 'vase',
    140: 'vegetable',
    141: 'vehicle',
    142: 'wave',
    143: 'wheel',
    144: 'window',
    145: 'windshield',
    146: 'wing',
    147: 'wire',
    148: 'woman',
    149: 'zebra'
}

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

if __name__=="__main__":
    rel_dict = {i-1: rel for i, rel in enumerate(CLASSES)}

    print(rel_dict)