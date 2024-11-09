import json
import logging
import random
import torch


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
    def __init__(self, glove, rels_json_file_path, split):
        logging.info(f"Reading the dataset at '{rels_json_file_path}'")
        self.file_path = rels_json_file_path
        f = open(rels_json_file_path, "r")
        js = json.load(f)

        self.train = []
        self.validate = []
        self.test = []
        self.pred_count = 0
        self.pred_ids = {}

        skipped = 0
        image_count = len(js)
        for i, pic in enumerate(js):
            for obj in pic["relationships"]:
                has_1 = glove.has(obj["object"]["name"])
                has_2 = glove.has(obj["subject"]["name"])
                if not has_1 or not has_2:
                    skipped += 1
                    continue

                (x, y) = self.parse_relationship(obj, glove)

                to_set = random.random()
                if to_set < split / 100:
                    self.test.append((x, y))
                elif to_set < 2 * split / 100:
                    self.validate.append((x, y))
                else:
                    self.train.append((x, y))

            if (i % 2000 == 0):
                logging.info(f"Read {i+1}/{image_count} images")

        f.close()

        logging.info("Finished reading the dataset")
        logging.info(f"Predicate count: {self.pred_count}")
        logging.info(f"Test set: {len(self.test)} rels")
        logging.info(f"Validation set: {len(self.validate)} rels")
        logging.info(f"Train set: {len(self.train)} rels")
        logging.info(f"Predicate count: {self.pred_count}")
        logging.info(f"Skipped due to not being in Glove: {skipped}")

    def parse_relationship(self, obj, glove):
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
            torch.tensor([o.x1, o.y1, o.size()[0], o.size()[1]]),
            subj_vec,
            torch.tensor([s.x1, s.y1, s.size()[0], s.size()[1]])
        ))

        if obj["predicate"] not in self.pred_ids:
            self.pred_ids[obj["predicate"]] = self.pred_count
            self.pred_count += 1
        y = self.pred_ids[obj["predicate"]]

        return x, y

    def train_set(self):
        return SubSet(self.train)

    def test_set(self):
        return SubSet(self.test)

    def validation_set(self):
        return SubSet(self.validate)

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


class SubSet:
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]
