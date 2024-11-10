import json
import logging
import torch
from torch.utils import data
import argparse
from glove import Glove
import sys


class Bounds:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

    @staticmethod
    def from_center_size(center_x, center_y, w, h):
        return Bounds(
            center_x - w * 0.5,
            center_y - h * 0.5,
            center_x + w * 0.5,
            center_y - h * 0.5,
        )

    @staticmethod
    def from_corner_size(x, y, w, h):
        return Bounds(x, y, x + w, y + h)

    def center(self):
        return [(self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5]

    def size(self):
        return [abs(self.x1 - self.x2), abs(self.y1 - self.y2)]

    def union(self, other):
        return Bounds(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            min(self.x2, other.x2),
            min(self.y2, other.y2),
        )

    def normalize(self, other):
        u = self.union(other)
        s = Bounds(
            (self.x1 - u.x1) / u.size()[0],
            (self.y1 - u.y1) / u.size()[1],
            (self.x2 - u.x1) / u.size()[0],
            (self.y2 - u.y1) / u.size()[1],
        )
        o = Bounds(
            (other.x1 - u.x1) / u.size()[0],
            (other.y1 - u.y1) / u.size()[1],
            (other.x2 - u.x1) / u.size()[0],
            (other.y2 - u.y1) / u.size()[1],
        )
        return [s, o]


class DataSet:
    def __init__(self, file_path, split, seed):
        logging.info(f"Reading the dataset at '{file_path}'")
        self.xs, self.ys, self.id_to_pred = torch.load(
            file_path, weights_only=True
        )
        self.pred_count = int(self.ys.max()+1)

        frac = split * 1.0 / 100.0
        self.train, self.test, self.validate = data.random_split(
            self,
            [1.0 - frac * 2.0, frac, frac],
            torch.Generator().manual_seed(seed),
        )
        logging.info(f"Predicate count: {self.pred_count}")
        logging.info(f"Size of training set: {len(self.train)}")
        logging.info(f"Size of validation set: {len(self.validate)}")
        logging.info(f"Size of test set: {len(self.test)}")

    def __getitem__(self, idx):
        return (self.xs[idx], self.ys[idx])

    def __len__(self):
        return len(self.xs)

    def get_predicate(self, idx):
        return self.id_to_pred[idx]

    def train_set(self):
        return SubSet(self.train)

    def test_set(self):
        return SubSet(self.test)

    def validation_set(self):
        return SubSet(self.validate)


class SubSet:
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]


class ProcessDataset:
    def __init__(self, path, glove, allowed_preds):
        self.glove = glove
        logging.info(f"Processing dataset '{path}'")
        self.pred_ids = {}
        self.id_to_pred = []

        f = open(allowed_preds, "r")
        for line in f:
            x = line.split(': ')
            main_pred = x[0].strip()
            self.pred_ids[main_pred] = len(self.id_to_pred)
            if len(x) > 1:
                for same_pred in x[1].split(', '):
                    self.pred_ids[same_pred] = len(self.id_to_pred)
            self.id_to_pred.append(main_pred)
        f.close()

        f = open(path, "r")
        js = json.load(f)
        skipped = 0
        image_count = len(js)
        objs = []
        for i, pic in enumerate(js):
            for obj in pic["relationships"]:
                ok_1 = glove.has(obj["object"]["name"])
                ok_2 = glove.has(obj["subject"]["name"])
                ok_3 = obj["predicate"] in self.pred_ids
                if not ok_1 or not ok_2 or not ok_3:
                    skipped += 1
                    continue

                (x, y) = self.parse_relationship(obj)
                objs.append((x, y))

            if (i % 2000 == 0):
                logging.info(f"Read {i+1}/{image_count} images")

        f.close()

        self.pred_count = len(self.id_to_pred)
        logging.info("Finished reading the dataset")
        logging.info(f"Predicate count: {self.pred_count}")
        logging.info(f"Skipped: {skipped * 100 / (len(objs)+skipped)}%")
        logging.info(f"Triple count: {len(objs)}")

        self.xs = torch.zeros((len(objs), len(objs[0][0])), dtype=torch.float)
        self.ys = torch.zeros(len(objs), dtype=torch.int64)
        for i, (x, y) in enumerate(objs):
            self.xs[i] = x
            self.ys[i] = y

    def parse_relationship(self, obj):
        obj_bounds = Bounds.from_corner_size(
            obj["object"]["x"],
            obj["object"]["y"],
            obj["object"]["w"],
            obj["object"]["h"],
        )
        subj_bounds = Bounds.from_corner_size(
            obj["subject"]["x"],
            obj["subject"]["y"],
            obj["subject"]["w"],
            obj["subject"]["h"],
        )
        o, s = obj_bounds.normalize(subj_bounds)

        obj_vec = glove.get(obj["object"]["name"])
        subj_vec = glove.get(obj["object"]["name"])
        x = torch.concat((
            obj_vec,
            torch.tensor([o.x1, o.y1, o.size()[0], o.size()[1]],
                         dtype=torch.float),
            subj_vec,
            torch.tensor([s.x1, s.y1, s.size()[0], s.size()[1]],
                         dtype=torch.float),
            torch.tensor([s.x1 - o.x1, s.y1 - o.y1]),
        ))

        y = self.pred_ids[obj["predicate"]]

        return x, y

    def save(self, res_path):
        torch.save([self.xs, self.ys, self.id_to_pred], res_path)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    help="Path to raw dataset relationships.json",
    required=True
)
parser.add_argument(
    "--output",
    help="Path to new processed dataset",
    required=True
)
parser.add_argument(
    "--glove",
    help="File path to glove.6B.50d.txt",
    required=True,
)
parser.add_argument(
    "--log",
    help="Log level",
    default="INFO"
)
parser.add_argument(
    "--allowed_preds",
    help="File with a list of allowed predicates",
    required=True,
)

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    logging.basicConfig(level=logging._nameToLevel[args.log])
    logging.info("Processing the raw dataset")
    glove = Glove(args.glove)
    ds = ProcessDataset(args.input, glove, args.allowed_preds)
    ds.save(args.output)


# {
#     'predicate': 'with',
#      'object': {
#         'name': 'windows',
#         'h': 148,
#         'object_id': 3798579,
#         'synsets': ['window.n.01'],
#         'w': 173,
#         'y': 4,
#         'x': 602
#     },
#     'relationship_id': 4265927,
#     'synsets': [],
#     'subject': {
#         'name': 'building',
#         'h': 536,
#         'object_id': 1058508,
#         'synsets': ['building.n.01'],
#         'w': 218,
#         'y': 2,
#         'x': 1,
#     }
# }
