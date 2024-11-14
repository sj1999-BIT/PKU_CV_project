import torch
import matplotlib.pyplot as plt
import argparse
import sys
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


class WordsFromFile:
    def __init__(self, file_path):
        self.word_to_id = {}
        self.id_to_word = []
        self.word_orig_id = {}
        self.orig_id_to_word = []

        with open(file_path, "r") as f:
            for line in f:
                x = str.split(line, ": ")
                main = str.strip(x[0])
                id = len(self.id_to_word)
                self.id_to_word.append(main)
                self.word_to_id[main] = id
                self.word_orig_id[main] = len(self.orig_id_to_word)
                self.orig_id_to_word.append(main)

                if len(x) == 1:
                    continue

                subs = str.split(x[1], ", ")
                for sub in subs:
                    sub = str.strip(sub)
                    self.word_to_id[sub] = id
                    self.word_orig_id[sub] = len(self.orig_id_to_word)
                    self.orig_id_to_word.append(sub)

    def word_idx(self, word):
        if word in self.word_to_id:
            return self.word_to_id[word]
        else:
            return None

    def get_word(self, idx):
        return self.id_to_word[idx]

    def word_orig_idx(self, word):
        if word in self.word_orig_id:
            return self.word_orig_id[word]
        else:
            return None

    def get_orig_word(self, orig_idx):
        return self.orig_id_to_word[orig_idx]


class AllWords:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = []

    def word_idx(self, word):
        if word in self.word_to_id:
            return self.word_to_id[word]
        else:
            id = len(self.id_to_word)
            self.word_to_id[word] = id
            self.id_to_word.append(word)
            return id

    def get_word(self, idx):
        return self.id_to_word[idx]

    def word_orig_idx(self, word):
        return self.word_idx(word)

    def get_orig_word(self, orig_idx):
        return self.get_word(orig_idx)


class Dataset:
    def __init__(self, args):
        logging.info(f"Reading the glove '{args.glove}' file")
        self.glove = Glove(args.glove)
        self.args = args

        if args.preds is not None:
            logging.info(f"Reading predicates from '{args.preds}'")
            self.preds = WordsFromFile(args.preds)
        else:
            logging.info(
                "No predicate filter file given, all predicates are accepted")
            self.preds = AllWords()

        if args.objs is not None:
            logging.info(f"Reading objects from '{args.objs}'")
            self.objs = WordsFromFile(args.objs)
        else:
            logging.info(
                "No objects filter file given, all objects are accepted")
            self.objs = AllWords()

        logging.info(f"Reading triples from '{args.input}'")
        if str.endswith(args.input, ".json"):
            logging.info("Parsing raw json...")
            self.parse_json()
        else:
            logging.info("Reading processed triples")
            self.triple_count, self.xs, self.ys, self.names = torch.load(
                args.input,
                weights_only=True
            )

        logging.info("Dataset constructed!")
        logging.info("Postprocessing")

    def parse_json(self):
        triples = {}
        skipped = 0

        with open(args.input, "r") as f:
            imgs = json.load(f)
            img_count = len(imgs)

            for i, img in enumerate(imgs):
                for triple in img["relationships"]:
                    t = None
                    try:
                        t = self.parse_triple(triple)
                    except Exception as e:
                        logging.error(f"[{e}] Could not parse triple '{
                                      triple}' in img {i}")

                    if t is None:
                        skipped += 1
                        continue

                    id = self.preds.word_idx(t[0])
                    if t[0] not in triples:
                        triples[id] = []
                    triples[id].append(t)

                if i % 10_000 == 0:
                    logging.info(f"Read {i+1}/{img_count} images...")

        logging.info("Processing triples...")
        self.process_data(triples)

    def process_data(self, triples):
        pred_counts = torch.zeros(len(triples), dtype=torch.int64)

        self.input_dim = None
        for (pred_id, pred_triples) in triples.items():
            if self.input_dim is None and len(pred_triples) != 0:
                self.input_dim = len(self.to_vecs(pred_triples[0])[0])

            assert pred_counts[pred_id] == 0
            pred_counts[pred_id] = len(pred_triples)

        triple_count = pred_counts.sum()
        self.xs = torch.zeros(
            (triple_count, self.input_dim),
            dtype=torch.float
        )
        self.ys = torch.zeros(triple_count, dtype=torch.int64)
        self.names = torch.zeros((triple_count, 6), dtype=torch.int64)

        i = 0
        for pred_id, pred_triples in triples.items():
            for triple in pred_triples:
                x, y, names = self.to_vecs(triple)
                self.xs[i] = x
                self.ys[i] = y
                self.names[i] = names
                i += 1

                if i % 10_000 == 0:
                    logging.info(f"Processed {i+1}/{triple_count} triples")

        self.triple_count = len(pred_counts)
        if self.args.output is not None:
            logging.info(f"Saving processed triples to '{self.args.output}'")
            torch.save(
                [self.triple_count, self.xs, self.ys, self.names],
                self.args.output
            )

    def to_vecs(self, triple):
        (pred_name, obj_name, subj_name, obj_bb, subj_bb) = triple

        o, s = Bounds.normalize(obj_bb, subj_bb)
        oc = torch.tensor(o.center())
        sc = torch.tensor(s.center())
        d = torch.sub(sc, oc)

        obj_main_name = self.objs.get_word(self.objs.word_idx(obj_name))
        subj_main_name = self.objs.get_word(self.objs.word_idx(subj_name))

        obj_vec = self.glove.get(obj_main_name)
        subj_vec = self.glove.get(subj_main_name)

        x = torch.concat((
            obj_vec,
            subj_vec,
            torch.concat([
                torch.concat((
                    torch.tensor([o.x1, o.y1, o.size()[0], o.size()[1]]),
                    torch.tensor([s.x1, s.y1, s.size()[0], s.size()[1]]),
                    oc,
                    sc,
                    d
                ))
            ] * self.args.repeat_bbs_count),
        ))
        y = self.preds.word_idx(pred_name)
        names = torch.tensor([
            self.preds.word_orig_idx(pred_name),
            self.preds.word_idx(pred_name),
            self.objs.word_orig_idx(obj_name),
            self.objs.word_idx(obj_name),
            self.objs.word_orig_idx(subj_name),
            self.objs.word_idx(subj_name),
        ], dtype=torch.int64)

        return x, y, names

    def parse_triple(self, triple):
        pred_name = triple["predicate"]
        obj_name = triple["object"]["name"]
        subj_name = triple["subject"]["name"]

        pred_id = self.preds.word_idx(pred_name)
        obj_id = self.objs.word_idx(obj_name)
        subj_id = self.objs.word_idx(subj_name)

        if pred_id is None or obj_id is None or subj_id is None:
            return None

        if self.glove.get(obj_name) is None or self.glove.get(subj_name) is None:
            return None

        obj_bb = Bounds.from_corner_size(
            int(triple["object"]["x"]),
            int(triple["object"]["y"]),
            int(triple["object"]["w"]),
            int(triple["object"]["h"]),
        )
        subj_bb = Bounds.from_corner_size(
            int(triple["subject"]["x"]),
            int(triple["subject"]["y"]),
            int(triple["subject"]["w"]),
            int(triple["subject"]["h"]),
        )

        return (
            pred_name,
            obj_name,
            subj_name,
            obj_bb,
            subj_bb,
        )

    def split(self, split, seed):
        pass


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    help="Path to the relationships.json file, or the processed triples file",
    required=True
)
parser.add_argument(
    "--output",
    help="If 'input' is a .json, saves the proccessed triples here. If not set, triples are not saved",
)
parser.add_argument(
    "--glove",
    help="Path to the glove.6B.50d.txt file",
    required=True
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
    "--stats",
    help="Directory for the output graphs, if not given graphs will not be generated",
)
parser.add_argument(
    "--log",
    help="Log level",
    default="INFO",
)
parser.add_argument(
    "--repeat_bbs_count",
    help="Repeat bounding boxes in the input vector n times",
    type=int,
    default=5,
)
parser.add_argument(
    "--obj_min",
    help="If the object is used in a smaller % of triples, drop it",
    type=float,
    default=0.006666667,
)
parser.add_argument(
    "--objs_output",
    help="Output for allowed objects file",
)
parser.add_argument(
    "--preds_output",
    help="Output for allowed predicates file",
)
parser.add_argument(
    "--obj_max",
    help="Drop triples if object is used in more than % of triples",
    type=float,
    default=100.0
)
parser.add_argument(
    "--pred_min",
    help="If the predicate is used in a smaller % of triples, drop it",
    type=float,
    default=0.006666667,
)
parser.add_argument(
    "--pred_max",
    help="Drop triples if predicate is used in more than % of triples",
    type=float,
    default=100.0
)
parser.add_argument(
    "--seed",
    help="Random seed",
    type=int,
    default=0,
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
