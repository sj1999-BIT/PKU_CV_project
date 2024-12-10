import torch
import matplotlib.pyplot as plt
import argparse
import sys
import os
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


class Group:
    def __init__(self):
        self.pred = None
        self.objs = None
        self.bbs = None
        self.img_name = None

    def __getitem__(self, idx):
        return [self.pred, self.objs, self.bbs, self.img_name][idx]


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

    def to_tensors(self):
        # len(class0), len(w0)..len(wn), len(class1)
        sizes = []
        str_bytes = []
        for words in self.classes:
            sizes.append(len(words))
            for w in words:
                sizes.append(len(w))
                for c in str.encode(w, "ascii"):
                    str_bytes.append(c)

        return torch.tensor(sizes), torch.tensor(str_bytes)

    @staticmethod
    def from_tensors(sizes, str_bytes):
        res = SynonimSet()
        cur_class = []
        class_len = 0
        str_start = 0
        for size in sizes:
            if class_len == 0:
                class_len = size

                if len(cur_class) > 0:
                    res.add_class(cur_class[0], cur_class[1:])
                    cur_class.clear()
            else:
                w = "".join(map(chr, str_bytes[str_start:str_start+size]))
                cur_class.append(w)
                str_start += size
                class_len -= 1

        if len(cur_class) > 0:
            res.add_class(cur_class[0], cur_class[1:])

        return res

    def remove_class(self, representative):
        id = self.word_to_class[representative][0]
        self.classes[id] = []
        self.unused.append(id)
        del self.word_to_class[representative]

    def get_repr(self, word):
        if word not in self.word_to_class:
            return None
        return self.classes[self.word_to_class[word][0]][0]

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

        logging.info(f"Reading from '{args.input}'")
        assert str.endswith(args.input, ".json")
        self.load_from_json()

        logging.info("Dataset constructed!")

    def load_from_json(self):
        self.preds = SynonimSet()
        self.objs = SynonimSet()

        if self.args.preds is not None:
            self.preds.load(self.args.preds)
        if self.args.objs is not None:
            self.objs.load(self.args.objs)

        triples, pred_counts, _ = self.parse_json()
        logging.info("Stage 2: Spliiting")
        groups = self.split(triples, pred_counts)

        logging.info("Stage 3: Dropping too common/uncommon")
        for i in range(len(groups)):
            triples = groups[i]
            logging.info(f"============Dropping in subset {i}============")
            pred_counts, obj_counts = self.count(triples)
            for round in range(self.args.round_limit):
                logging.info(
                    f"Round {round}/{self.args.round_limit} of dropping groups")
                triples, pred_counts, obj_counts, change = self.drop_triples(
                    triples, pred_counts, obj_counts)
                groups[i] = triples
                if change == 0:
                    break

        logging.info("Finished dropping")

        if not os.path.exists(self.args.output):
            os.mkdir(self.args.output)
        logging.info("Stage 3: Saving subsets")
        self.groups = []
        _preds = []
        _objs = []
        for i in range(len(groups)):
            logging.info(f"============Subset {i}============")
            pred_counts, obj_counts = self.count(groups[i])
            group, preds, objs = self.process_triples(
                groups[i], pred_counts, obj_counts)
            j = self.write_json(group)
            with open(f"{self.args.output}/rels_{i}.json", "w") as f:
                f.write(j)
            self.groups.append(group)
            _preds.append(preds)
            _objs.append(objs)

        self.preds = _preds
        self.objs = _objs

    def write_json(self, group):
        imgs = {}
        for g in group:
            if g.img_name not in imgs:
                imgs[g.img_name] = []
            imgs[g.img_name].append(g)

        imgs = [x for x in imgs.items()]
        res = {}
        for img_idx, (name, gs) in enumerate(imgs):
            res[name] = []
            for j, g in enumerate(gs):
                res[name].append({"predicate": g[0], "object": []})
                for i in range(len(g[1])):
                    res[name][j]["object"].append({
                       "name": g[1][i],
                       "x": g[2][i].x1,
                       "y": g[2][i].y1,
                       "w": g[2][i].size()[0],
                       "h": g[2][i].size()[1],
                    })
        return json.dumps(res, indent=4)

    def split(self, groups, pred_count):
        s = sorted(pred_count.items(), key=lambda x: -x[1])
        pred_split = []
        i = 0
        stop = False
        while (i < len(s) and not stop):
            biggest = s[i][1]
            g = []
            for j in range(i, len(s)):
                x = s[j][1]
                if biggest / x < self.args.split_factor:
                    g.append(s[j])
                else:
                    i = j
                    break
            else:
                stop = True
            pred_split.append(g)

        logging.info(f"Split into {len(pred_split)} subsets")
        split = {}
        new_groups = []
        for i, g in enumerate(pred_split):
            new_groups.append([])
            for p in g:
                split[p[0]] = i

        for g in groups:
            idx = split[g.pred]
            new_groups[idx].append(g)

        return new_groups

    def process_triples(self, triples, pred_counts, obj_counts):
        logging.info(f"==== Triple count: {len(triples)}")
        logging.info(f"==== Predicate count: {len(pred_counts)}")
        logging.info(f"==== Object count: {len(obj_counts)}")

        new_preds = SynonimSet()
        new_objs = SynonimSet()
        for word, _ in pred_counts.items():
            new_preds.add_class(word, self.preds.get_synonims(word))
        for word, _ in obj_counts.items():
            new_objs.add_class(word, self.objs.get_synonims(word))

        return triples, new_preds, new_objs

    def parse_json(self):
        logging.info("Stage 1: reading relationships")
        skipped = 0
        groups = []

        with open(args.input, "r") as f:
            imgs = json.load(f)
            img_count = len(imgs)

            for i, img in enumerate(imgs):
                for j, group in enumerate(imgs[img]):
                    t = None
                    try:
                        t = self.parse_group(group)
                    except Exception as e:
                        logging.error(f"[{e}] Could not parse group '{
                                      group}' in img {img}")
                    if t is None:
                        skipped += 1
                        continue
                    t.img_name = img
                    groups.append(t)
                if i % 10_000 == 0 or i + 1 == img_count:
                    logging.info(f"{i+1}/{img_count} images read...")

        pred_counts, obj_counts = self.count(groups)
        logging.info(f"Read {len(groups)} triples, skipped {skipped}")
        logging.info(f"==== {len(pred_counts)} predicates")
        logging.info(f"==== {len(obj_counts)} objects")

        return groups, pred_counts, obj_counts

    def drop_triples(self, triples, pred_counts, obj_counts):
        change = 0
        logging.info("Stage 3.1: Limiting too common predicates")
        new_triples, pred_counts, obj_counts = self.pred_upper_limit(
            triples,
            pred_counts,
            self.args.pred_max
        )
        change += len(triples)-len(new_triples)
        logging.info(f"Dropped {len(triples)-len(new_triples)} triples")
        triples = new_triples
        assert len(triples) > 0

        logging.info("Stage 3.2: Limiting too common objects")
        new_triples, pred_counts, obj_counts = self.obj_upper_limit(
            triples,
            obj_counts,
            self.args.obj_max
        )
        change += len(triples)-len(new_triples)
        logging.info(f"Dropped {len(triples)-len(new_triples)} triples")
        triples = new_triples
        assert len(triples) > 0

        logging.info("Stage 3.3: Dropping triples with uncommon predicates")
        new_triples, pred_counts, obj_counts = self.pred_lower_limit(
            triples,
            pred_counts,
            self.args.pred_min
        )
        change += len(triples)-len(new_triples)
        logging.info(f"Dropped {len(triples)-len(new_triples)} triples")
        triples = new_triples
        assert len(triples) > 0

        logging.info("Stage 3.4: Dropping triples with uncommon objects")
        new_triples, pred_counts, obj_counts = self.obj_lower_limit(
            triples,
            obj_counts,
            self.args.pred_min
        )
        change += len(triples)-len(new_triples)
        logging.info(f"Dropped {len(triples)-len(new_triples)} triples")
        triples = new_triples
        assert len(triples) > 0
        logging.info(f"Predicates: {len(pred_counts)}:{
                     sum(pred_counts.values())}")
        logging.info(f"Objects: {len(obj_counts)}:{sum(obj_counts.values())}")

        return new_triples, pred_counts, obj_counts, change

    def pred_lower_limit(self, triples, counts, low):
        dropped = []
        for word, count in counts.items():
            if low is not None and count < low:
                dropped.append(word)

        dropped = set(dropped)
        new_triples = []
        for t in triples:
            if t[0] in dropped:
                continue
            new_triples.append(t)

        new_pred_counts, new_obj_counts = self.count(new_triples)
        return new_triples, new_pred_counts, new_obj_counts

    def obj_lower_limit(self, triples, counts, low):
        dropped = []
        for word, count in counts.items():
            if low is not None and count < low:
                dropped.append(word)

        dropped = set(dropped)
        new_triples = []
        for t in triples:
            objs = []
            bbs = []
            for i, w in enumerate(t[1]):
                if w in dropped:
                    continue
                objs.append(w)
                bbs.append(t[2][i])
            t.objs = objs
            t.bbs = bbs 
            if len(objs) > 1:
                new_triples.append(t)

        new_pred_counts, new_obj_counts = self.count(new_triples)
        return new_triples, new_pred_counts, new_obj_counts

    def pred_upper_limit(self, triples, counts, top):
        drop_chance = {}
        for word, count in counts.items():
            if top is not None and count > top:
                drop_chance[word] = top / count

        new_triples = []
        for t in triples:
            if t[0] in drop_chance and torch.rand((1), generator=self.rng) < drop_chance[t[0]]:
                continue
            new_triples.append(t)

        new_pred_counts, new_obj_counts = self.count(new_triples)
        return new_triples, new_pred_counts, new_obj_counts

    def obj_upper_limit(self, triples, counts, top):
        drop_chance = {}
        for word, count in counts.items():
            if top is not None and count > top:
                drop_chance[word] = top / count

        new_triples = []
        count = 0
        for t in triples:
            objs = []
            bbs = []
            for i, w in enumerate(t[1]):
                if w in drop_chance and torch.rand((1), generator=self.rng) < drop_chance[w]:
                    continue
                objs.append(w)
                bbs.append(t[2][i])
            t.objs = objs
            t.bbs = bbs
            if len(t.objs) > 1:
                new_triples.append(t)

        new_pred_counts, new_obj_counts = self.count(new_triples)
        return new_triples, new_pred_counts, new_obj_counts

    def count(self, groups):
        pred_counts = {}
        obj_counts = {}
        for g in groups:
            if g[0] not in pred_counts:
                pred_counts[g[0]] = 0
            pred_counts[g[0]] += 1
            for o in g[1]:
                if o not in obj_counts:
                    obj_counts[o] = 0
                obj_counts[o] += 1

        return pred_counts, obj_counts

    def parse_group(self, group_json):
        g = Group()
        g.pred = str.lower(group_json["predicate"])
        self.preds.add_class(g.pred)
        g.pred = self.preds.get_repr(g.pred)
        if g.pred is None:
            return None

        objects = []
        bbs = []
        for i, obj in enumerate(group_json["object"]):
            obj_name = obj["name"]
            self.objs.add_class(obj_name)
            obj_name = self.objs.get_repr(obj_name)
            if obj_name is None or self.glove.get(obj_name) is None:
                continue
            x = float(obj["x"])
            y = float(obj["y"])
            w = float(obj["w"])
            h = float(obj["h"])
            objects.append(obj_name)
            bbs.append(Bounds.from_corner_size(x, y, w, h))

        g.objs = objects
        g.bbs = bbs
        return g

    def draw_predicates(self):
        x = []
        y = []
        for i in range(len(self.groups)):
            start = len(x)
            for j in range(self.preds[i].class_count()):
                x.append("")
                y.append(0)
            cnt, _ = self.count(self.groups[i])
            for w, n in cnt.items():
                idx = start + self.preds[i].get_class_id(w)
                x[idx] = f"{w}:{i}"
                y[idx] = n
            x.append(f"=={i}==")
            y.append(0)

        plt.title("Frequency of predicates")
        plt.ylabel("Number of triples")
        plt.tick_params(axis="x", labelrotation=90)
        plt.bar(x, y)
        plt.show()

    def draw_objects(self):
        x = []
        y = []
        for i in range(len(self.groups)):
            start = len(x)
            for j in range(self.objs[i].class_count()):
                x.append("")
                y.append(0)
            _, cnt = self.count(self.groups[i])
            for w, n in cnt.items():
                idx = start + self.objs[i].get_class_id(w)
                x[idx] = f"{w}:{i}"
                y[idx] = n
            x.append(f"=={i}==")
            y.append(0)

        plt.title("Frequency of Objects")
        plt.ylabel("Number of triples")
        plt.tick_params(axis="x", labelrotation=90)
        plt.bar(x, y)
        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    help="Path to the relationships.json file, or the processed triples file",
    required=True
)
parser.add_argument(
    "--output",
    help="Output dir for the split dataset",
    required=True
)
parser.add_argument(
    "--split_factor",
    help="Split the data by predicates into groups, where most_used/least_used < split_factor",
    default=5.0,
    type=float
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
    ds.draw_predicates()
    ds.draw_objects()
