import torch
import matplotlib.pyplot as plt
import argparse
import sys
import os
import shutil
import logging
import json


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

    def div(self, x, y):
        self.x1 /= x
        self.x2 /= x
        self.y1 /= y
        self.y2 /= y

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


class Glove:
    def __init__(self, glove_file_path):
        f = open(glove_file_path, "r")
        self.word_to_idx = {}
        self.mat = []
        for i, line in enumerate(f):
            word, *vec = line.split()
            self.word_to_idx[word] = i
            self.mat.append(torch.tensor(
                [float(x) for x in vec],
                dtype=torch.float)
            )

        f.close()

    def get(self, word):
        if word in self.word_to_idx:
            return self.mat[self.word_to_idx.get(word)]
        else:
            return None


class Triple:
    def __init__(self):
        self.pred = None
        self.obj = None
        self.subj = None
        self.obb = None
        self.sbb = None
        self.img_id = None
        self.img_path = None

    def __getitem__(self, idx):
        return [self.pred, self.obj, self.subj, self.obb, self.sbb, self.img_id, self.img_path][idx]


class SynonimSet:
    def __init__(self):
        self.word_to_class = {}
        self.classes = []
        self.unused = []
        self.locked = False

    def load(self, file):
        with open(file, "r") as file:
            for line in file:
                x = str.split(line, ": ")
                repr = str.strip(x[0])
                syns = []
                if len(x) > 1:
                    for s in str.split(x[1], ", "):
                        syns.append(str.strip(s))
                self.add_class(repr, syns)
            self.locked = True

    def class_count(self):
        return len(self.classes) - len(self.unused)

    def representatives(self):
        res = []
        for c in self.classes:
            if len(c) > 0:
                res.append(c[0])
        return res

    def union(self, word1, word2):
        c1 = self.get_class_id(word1)
        c2 = self.get_class_id(word2)
        if c1 == c2:
            return
        for w in self.classes[c2]:
            self.word_to_class[w] = (c1, len(self.classes[c1]))
            self.classes[c1].append(w)
        self.classes[c2] = []
        self.unused.append(c2)

    def add_class(self, representative, synonims=[]):
        if self.locked:
            return

        self.add_word(representative)
        id = self.get_class_id(representative)
        for s in synonims:
            if s in self.word_to_class:
                self.union(representative, s)
            else:
                self.word_to_class[s] = (id, len(self.classes[id]))
                self.classes[id].append(s)

    def add_word(self, word):
        if word in self.word_to_class:
            return
        id = 0
        if len(self.unused) > 0:
            id = self.unused.pop(0)
        else:
            id = len(self.classes)
            self.classes.append([])

        self.classes[id].append(word)
        self.word_to_class[word] = (id, 0)
        return

    def save(self, dir, name):
        with open(f"{dir}/{name}", "w") as f:
            for words in self.classes:
                if len(words) == 1:
                    f.write(f"{words[0]}\n")
                else:
                    f.write(f"{words[0]}: ")
                    for i, w in enumerate(words[1:]):
                        f.write(f"{w}")
                        if i + 1 != len(words[1:]):
                            f.write(", ")
                        else:
                            f.write("\n")

    def remove_class(self, representative):
        id = self.word_to_class[representative][0]
        self.classes[id] = []
        self.unused.append(id)
        del self.word_to_class[representative]

    def get_repr(self, word):
        if word not in self.word_to_class:
            return None
        return self.classes[self.word_to_class[word][0]][0]

    def get_repr_by_id(self, class_id):
        return self.classes[class_id][0]

    def get_synonims(self, word):
        return self.classes[self.word_to_class[word][0]][1:]

    def get_class_id(self, word):
        return self.word_to_class[word][0]

    def get_word_id(self, word):
        return self.word_to_class[word][1]


class Dataset:
    def __init__(self, args):
        logging.info(f"Reading the glove '{args.glove}' file")
        self.rng = torch.Generator().manual_seed(args.seed)
        self.glove = Glove(args.glove)
        self.args = args

        logging.info(f"Reading triples from '{args.relationships}'")
        assert str.endswith(args.relationships, ".json")
        self.preds = SynonimSet()
        self.objs = SynonimSet()

        if self.args.preds is not None:
            self.preds.load(self.args.preds)
        if self.args.objs is not None:
            self.objs.load(self.args.objs)

        logging.info("Stage 1: reading raw triples")
        triples, pred_counts, obj_counts = self.parse_json()

        logging.info("Stage 3: Dropping triples")
        for round in range(self.args.round_limit):
            logging.info(
                f"Round {round}/{self.args.round_limit} of dropping triples")
            triples, pred_counts, obj_counts, change = self.drop_triples(
                triples, pred_counts, obj_counts)
            if len(triples) == 0:
                logging.warning("Triples empty!")
            if change == 0:
                break

        logging.info("Stage 3: Splitting triples")
        subsets = self.split(triples, pred_counts)
        logging.info(f"Split into {len(subsets)} subsets")

        logging.info("Stage 4: Converting to YOLO format")
        if not os.path.exists(self.args.output):
            os.mkdir(self.args.output)

        for i in range(len(subsets)):
            self.convert(i, subsets[i])

        logging.info("Dataset converted!")

    def convert(self, i, triples):
        if len(triples) == 0:
            logging.warning(f"Triple subset {i} is empty")
            return
        prefix = f"{self.args.output}/subset_{i}"
        logging.info(f"Converting subset {i}, saved in {prefix}")
        if not os.path.exists(prefix):
            os.mkdir(prefix)

        pred_counts, obj_counts = self.count_triples(triples)
        new_preds = SynonimSet()
        new_objs = SynonimSet()
        for word, _ in pred_counts.items():
            new_preds.add_class(word, self.preds.get_synonims(word))
        for word, _ in obj_counts.items():
            new_objs.add_class(word, self.objs.get_synonims(word))

        logging.info("Writing object names...")
        with open(f"{prefix}/obj_labels.txt", "w") as f:
            for c in range(new_objs.class_count()):
                f.write(f"{new_objs.get_repr_by_id(c)}\n")

        logging.info("Writing predicate names...")
        with open(f"{prefix}/pred_labels.txt", "w") as f:
            for c in range(new_preds.class_count()):
                f.write(f"{new_preds.get_repr_by_id(c)}\n")

        logging.info("Writing objects and triples, copy image...")
        if not os.path.exists(f"{prefix}/obj_labels"):
            os.mkdir(f"{prefix}/obj_labels")
        if not os.path.exists(f"{prefix}/pred_labels"):
            os.mkdir(f"{prefix}/pred_labels")

        if not os.path.exists(f"{prefix}/images"):
            os.mkdir(f"{prefix}/images")

        cur_id = -1
        obj_file = None
        rel_file = None
        obj_idx_in_file = {}

        triples.sort(key=lambda t: t.img_id)

        triple_count = len(triples)
        percent_step = 5
        prev_percent = -percent_step
        for i, t in enumerate(triples):
            assert cur_id <= t.img_id
            if cur_id != t.img_id:
                cur_id = t.img_id
                if obj_file is not None:
                    obj_file.close()
                if rel_file is not None:
                    rel_file.close()
                (folder, name) = t.img_path
                (name, ext) = str.split(name, ".")
                obj_file = open(f"{prefix}/obj_labels/{folder}_{name}.txt", "w")
                rel_file = open(f"{prefix}/pred_labels/{folder}_{name}.txt", "w")
                if cur_id not in obj_idx_in_file:
                    obj_idx_in_file[cur_id] = {"next": 0, "ids": {}}
                shutil.copyfile(f"{self.args.vg}/{folder}/{name}.{ext}", f"{prefix}/images/{folder}_{name}.{ext}")
                

            obj_global_idx = new_objs.get_class_id(t.obj)
            if (obj_global_idx, t.obb) not in obj_idx_in_file[cur_id]["ids"]:
                x, y = t.obb.center()
                w, h = t.obb.size()
                obj_file.write(f"{obj_global_idx} {x} {y} {w} {h}\n")
                obj_idx_in_file[cur_id]["ids"][(obj_global_idx, t.obb)] = obj_idx_in_file[cur_id]["next"]
                obj_idx_in_file[cur_id]["next"] += 1
            obj_local_idx = obj_idx_in_file[cur_id]["ids"][(obj_global_idx, t.obb)]

            subj_global_idx = new_objs.get_class_id(t.subj)
            if (subj_global_idx, t.sbb) not in obj_idx_in_file[cur_id]["ids"]:
                x, y = t.sbb.center()
                w, h = t.sbb.size()
                obj_file.write(f"{subj_global_idx} {x} {y} {w} {h}\n")
                obj_idx_in_file[cur_id]["ids"][(subj_global_idx, t.sbb)] = obj_idx_in_file[cur_id]["next"]
                obj_idx_in_file[cur_id]["next"] += 1
            subj_global_idx = obj_idx_in_file[cur_id]["ids"][(subj_global_idx, t.sbb)]

            pred_idx = new_preds.get_class_id(t.pred)

            rel_file.write(f"{obj_local_idx} {subj_global_idx} {pred_idx}\n")

            if (i * 100) // triple_count >= prev_percent + percent_step:
                prev_percent = (i * 100) // triple_count
                logging.info(f"{prev_percent}% done...")

    def process_triples(self, triples, pred_counts, obj_counts):
        logging.info("Stage 3: Processing triples")
        logging.info(f"==== Triple count: {len(triples)}")
        logging.info(f"==== Predicate count: {len(pred_counts)}")
        logging.info(f"==== Object count: {len(obj_counts)}")

        new_preds = SynonimSet()
        new_objs = SynonimSet()
        for word, _ in pred_counts.items():
            new_preds.add_class(word, self.preds.get_synonims(word))
        for word, _ in obj_counts.items():
            new_objs.add_class(word, self.objs.get_synonims(word))

        self.preds = new_preds
        self.objs = new_objs
        self.xs = torch.zeros(
            (len(triples), self.input_size()),
            dtype=torch.float
        )
        self.ys = torch.zeros(len(triples), dtype=torch.int64)
        self.names = torch.zeros((len(triples), 6), dtype=torch.int64)
        self.imgs = torch.zeros((len(triples), 2), dtype=torch.int64)

        for i, t in enumerate(triples):
            x, y, names, img = t.to_vec(
                self.glove, self.preds, self.objs, self.args.repeat_bbs_count)
            self.xs[i] = x
            self.ys[i] = y
            self.names[i] = names
            self.imgs[i] = img

            if i % 100_000 == 0:
                logging.info(f"{i+1}/{len(triples)} triples processed..")

    def parse_json(self):
        skipped = 0
        triples = []

        self.image_data = []
        with open(self.args.image_data, "r") as f:
            self.image_data = json.load(f)

        with open(self.args.relationships, "r") as f:
            imgs = json.load(f)
            img_count = len(imgs)

            percent_step = 1
            prev_percent = -percent_step
            for i, img in enumerate(imgs):
                for j, triple in enumerate(img["relationships"]):
                    t = None
                    try:
                        t = self.parse_triple(triple)
                    except Exception as e:
                        logging.error(f"[{e}] Could not parse triple '{
                                      triple}' in img {i}")
                    if t is None:
                        skipped += 1
                        continue
                    t.obb.div(self.image_data[i]["width"], self.image_data[i]["height"])
                    t.sbb.div(self.image_data[i]["width"], self.image_data[i]["height"])

                    _, folder, name = str.rsplit(self.image_data[i]["url"], "/", maxsplit=2)
                    t.img_id = img["image_id"]
                    t.img_path = (folder, name)
                    assert self.image_data[i]["image_id"] == t.img_id
                    triples.append(t)
                if (i * 100) // img_count >= prev_percent + percent_step:
                    prev_percent = (i * 100) // img_count
                    logging.info(f"[{prev_percent}%] {i+1}/{img_count} images read...")

                if self.args.image_limit is not None and i + 1 == self.args.image_limit:
                    break

        pred_counts, obj_counts = self.count_triples(triples)
        logging.info(f"Read {len(triples)} triples, skipped {skipped}")
        logging.info(f"==== {len(pred_counts)} predicates")
        logging.info(f"==== {len(obj_counts)} objects")

        return triples, pred_counts, obj_counts

    def drop_triples(self, triples, pred_counts, obj_counts):
        change = 0
        logging.info("Stage 2.1: Limiting too common predicates")
        new_triples, pred_counts, obj_counts = self.upper_limit(
            triples,
            0,
            pred_counts,
            self.args.pred_max
        )
        change += len(triples)-len(new_triples)
        logging.info(f"Dropped {len(triples)-len(new_triples)} triples")
        triples = new_triples

        logging.info("Stage 2.2: Limiting too common objects")
        new_triples, pred_counts, obj_counts = self.upper_limit(
            triples,
            1,
            obj_counts,
            self.args.obj_max
        )
        change += len(triples)-len(new_triples)
        logging.info(f"Dropped {len(triples)-len(new_triples)} triples")
        triples = new_triples

        logging.info("Stage 2.3: Limiting too common subjects")
        new_triples, pred_counts, obj_counts = self.upper_limit(
            triples,
            2,
            obj_counts,
            self.args.obj_max
        )
        change += len(triples)-len(new_triples)
        logging.info(f"Dropped {len(triples)-len(new_triples)} triples")
        triples = new_triples

        logging.info("Stage 2.4: Dropping triples with uncommon predicates")
        new_triples, pred_counts, obj_counts = self.lower_limit(
            triples,
            0,
            pred_counts,
            self.args.pred_min
        )
        change += len(triples)-len(new_triples)
        logging.info(f"Dropped {len(triples)-len(new_triples)} triples")
        triples = new_triples

        logging.info("Stage 2.5: Dropping triples with uncommon objects")
        new_triples, pred_counts, obj_counts = self.lower_limit(
            triples,
            1,
            obj_counts,
            self.args.pred_min
        )
        change += len(triples)-len(new_triples)
        logging.info(f"Dropped {len(triples)-len(new_triples)} triples")
        triples = new_triples

        logging.info("Stage 2.6: Dropping triples with uncommon subjects")
        new_triples, pred_counts, obj_counts = self.lower_limit(
            triples,
            2,
            obj_counts,
            self.args.pred_min
        )
        change += len(triples)-len(new_triples)
        logging.info(f"Dropped {len(triples)-len(new_triples)} triples")
        triples = new_triples

        return new_triples, pred_counts, obj_counts, change

    def lower_limit(self, triples, idx, counts, low):
        dropped = []
        for word, count in counts.items():
            if low is not None and count < low:
                dropped.append(word)

        dropped = set(dropped)
        new_triples = []
        for t in triples:
            if t[idx] in dropped:
                continue
            new_triples.append(t)

        new_pred_counts, new_obj_counts = self.count_triples(new_triples)
        return new_triples, new_pred_counts, new_obj_counts

    def upper_limit(self, triples, idx, counts, top):
        drop_chance = {}
        for word, count in counts.items():
            if top is not None and count > top:
                drop_chance[word] = top / count

        new_triples = []
        for t in triples:
            if t[idx] in drop_chance and torch.rand((1), generator=self.rng) < drop_chance[t[idx]]:
                continue
            new_triples.append(t)

        new_pred_counts, new_obj_counts = self.count_triples(new_triples)
        return new_triples, new_pred_counts, new_obj_counts

    def count_triples(self, triples):
        pred_counts = {}
        obj_counts = {}
        for t in triples:
            if t[0] not in pred_counts:
                pred_counts[t[0]] = 0
            if t[1] not in obj_counts:
                obj_counts[t[1]] = 0
            if t[2] not in obj_counts:
                obj_counts[t[2]] = 0
            pred_counts[t[0]] += 1
            obj_counts[t[1]] += 1
            obj_counts[t[2]] += 1

        return pred_counts, obj_counts

    def parse_triple(self, triple_json):
        pred_name = str.lower(triple_json["predicate"])
        obj_name = str.lower(triple_json["object"]["name"])
        subj_name = str.lower(triple_json["subject"]["name"])

        self.preds.add_class(pred_name)
        self.objs.add_class(obj_name)
        self.objs.add_class(subj_name)

        t = Triple()
        t.pred = self.preds.get_repr(pred_name)
        t.obj = self.objs.get_repr(obj_name)
        t.subj = self.objs.get_repr(subj_name)

        if t.pred is None or t.obj is None or t.subj is None:
            return None

        if self.glove.get(t.obj) is None or self.glove.get(t.subj) is None:
            return None

        t.obb = Bounds.from_corner_size(
            float(triple_json["object"]["x"]),
            float(triple_json["object"]["y"]),
            float(triple_json["object"]["w"]),
            float(triple_json["object"]["h"]),
        )
        t.sbb = Bounds.from_corner_size(
            float(triple_json["subject"]["x"]),
            float(triple_json["subject"]["y"]),
            float(triple_json["subject"]["w"]),
            float(triple_json["subject"]["h"]),
        )

        return t

    def split(self, triples, pred_count):
        sorted_preds = sorted(pred_count.items(), key=lambda x: -x[1])
        pred_to_subset = {}

        start = 0
        subset = 0
        for i, (w, cnt) in enumerate(sorted_preds):
            if sorted_preds[start][1] / sorted_preds[i][1] > self.args.split_factor:
                start = i
                subset += 1
            pred_to_subset[sorted_preds[i][0]] = subset
        print(pred_to_subset)

        subsets = [[] for _ in range(subset + 1)]
        for t in triples:
            subsets[pred_to_subset[t[0]]].append(t)

        return subsets

parser = argparse.ArgumentParser()
parser.add_argument(
    "--relationships",
    help="Path to the relationships.json file",
    required=True
)
parser.add_argument(
    "--image_data",
    help="Path to the image_data.json file",
    required=True
)
parser.add_argument(
    "--vg",
    help="Path to the vg images (folder with VG_100K and VG_100K_2)",
    required=True,
)
parser.add_argument(
    "--output",
    help="Directory to output YOLO format files",
    required=True
)
parser.add_argument(
    "--round_limit",
    help="Limit the number of rounds of dropping triples",
    default=100,
    type=int,
)
parser.add_argument(
    "--glove",
    help="Path to the glove.6B.50d.txt file",
)
parser.add_argument(
    "--preds",
    help="Predicate filter file, if not given all predicates are used as-is",
)
parser.add_argument(
    "--objs",
    help="Object names filter file, if not given all object names are used as-is",
)
parser.add_argument(
    "--log",
    help="Log level",
    default="INFO",
)
parser.add_argument(
    "--obj_min",
    help="If the object is used less than obj_min times, drop triples with it",
    type=int,
    default=100,
)
parser.add_argument(
    "--obj_max",
    help="Drop triples if an object is used more than obj_max times. If not set, do not drop triples",
    type=int,
    default=300_000
)
parser.add_argument(
    "--pred_min",
    help="If the predicate is used less than pred_min times, drop triples with it",
    type=int,
    default=100,
)
parser.add_argument(
    "--pred_max",
    help="Drop triples if an predicate is used more than pred_max times. If not set, do not drop triples",
    type=int,
    default=300_000
)
parser.add_argument(
    "--seed",
    help="Random seed",
    type=int,
    default=0,
)
parser.add_argument(
    "--image_limit",
    help="Limit the number of images read (used for testing)",
    type=int,
)
parser.add_argument(
    "--split_factor",
    help="Split the dataset by predicate usage, where pred_max/pred_min < split_factor",
    type=float,
    default=5.0,
)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    logging.basicConfig(
        level=logging._nameToLevel[args.log],
        stream=sys.stdout,
        filemode="w",
        format="[{levelname}] {asctime}: {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    ds = Dataset(args)
