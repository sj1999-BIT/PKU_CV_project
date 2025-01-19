import torch
import argparse
import sys
from PredPred.src.dataset import Dataset
from PredPred.src.glove import Glove
from PredPred.src.model import Model
import json
import numpy as np
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models",
    help="Path to the models.json file",
    required=True,
)
parser.add_argument(
    "--glove",
    help="Path to glove.6B.50d.txt",
    required=True,
)
parser.add_argument(
    "--weights",
    help="Path to weights",
    required=True,
)
parser.add_argument(
    "--datasets",
    help="Path to out",
    required=True,
)
parser.add_argument(
    "--input",
    help="Path to input.json",
)
parser.add_argument(
    "--dir",
    help="Path to a bunch of files"
)
parser.add_argument(
    "--device",
    help="Device for running model on",
    default="cpu"
)

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

    def to_array(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])


class Runner:
    def __init__(self, args):
        self.args = args
        self.glove = Glove(args.glove)
        self.m = []
        
        with open(args.models) as f:
            self.models = json.load(f)

        self.cnt = 5
        self.input_size = 2 + 50 + 50 + (4 + 4 + 2 + 2 + 2) * self.cnt
        self.all_preds = []
        for i, m in enumerate(self.models["params"]):
            model = Model(
                self.input_size,
                m["layer_1"], 
                m["layer_2"], 
                m["layer_3"], 
                len(self.models["subsets"][i]),
                m["dropout"]
            )
            [_, _, objs, _] = torch.load(f"{args.datasets}/{m["subset"]}/dataset.bin", weights_only=True)
            self.m.append({
                "model": model,
                "preds": self.models["subsets"][i],
                "objs": objs,
                "idx": i,
            })
            model.load_state_dict(torch.load(
                f"{args.weights}/{m["subset"]}/model_{m["epoch_count"]-1:05d}.pth",
                weights_only=True,
                map_location=torch.device('cpu')
            ))
            model = model.to(torch.device('cpu'))
            model.eval()

            for p in self.models["subsets"][i]:
                self.all_preds.append(p)


    def eval(self, obj, obb, subj, sbb, pred):
        with torch.no_grad():
            model = None
            for m in self.m:
                if pred not in m["preds"]:
                    continue
                model = m
                break
            assert model is not None
            
            oid = None
            sid = None
            for (i, w) in enumerate(model["objs"]):
                if w == obj:
                    oid = i
                if w == subj:
                    sid = i
            if oid is None:
                print("WARNING no oid for", obj)
                oid = 0
            if sid is None:
                print("WARNING no sid for", subj)
                sid = 0

            obj_vec = self.glove.get(obj)
            subj_vec = self.glove.get(subj)
            obb, sbb = Bounds.normalize(obb, sbb)
            oc = np.array(obb.center())
            sc = np.array(sbb.center())
            diff = sc - oc

            x = torch.concat([
                torch.tensor([oid], dtype=torch.float),
                torch.tensor([sid], dtype=torch.float),
                obj_vec,
                subj_vec,
                torch.from_numpy(np.repeat(np.concatenate([
                    obb.to_array(),
                    sbb.to_array(),
                    oc,
                    sc,
                    diff
                ], dtype=np.float32), self.cnt))
            ])
            x = x.reshape((1, self.input_size))

            x = x.to(torch.device('cpu'))
            y = model["model"](x)
            y = y.to("cpu")
            y -= y.min()
            y /= y.max()
            return self.make_y(y[0], model["idx"])

    def make_y(self, y, susbset):
        res = torch.zeros(len(self.all_preds))
        offset = 0
        for i in range(0, susbset):
            offset += len(self.m[i]["preds"])

        for i in range(0, len(self.m[susbset]["preds"])):
            res[i + offset] = y[i]

        return res

    def run_single_image(self, name, img):
        runner = self
        res = []
        for group in img:
            pred = group["predicate"]
            idx = None
            highest = None
            for (i, w) in enumerate(runner.all_preds):
                if w == pred:
                    idx = i 
                    break
            assert idx is not None

            for obj in group["object"]:
                obb = Bounds.from_corner_size(obj["x"], obj["y"], obj["w"], obj["h"])
                for subj in group["object"]:
                    if obj == subj:
                        continue
                    sbb = Bounds.from_corner_size(subj["x"], subj["y"], subj["w"], subj["h"])
                    prob = runner.eval(
                            obj["name"],
                            obb,
                            subj["name"],
                            sbb,
                            pred
                    )[idx]
                    if highest is None or highest[0] < prob.item():
                        highest = (prob, obj["name"], obb, subj["name"], sbb)

            if highest is None:
                print("WARNING could not find a triple...")
                continue
            (prob, obj, obb, subj, sbb) = highest
            res.append({
                    "predicate": pred,
                    "object": { "name": obj, "x": obb.x1, "y": obb.y1, "w": obb.size()[0], "h": obb.size()[1]},
                    "subject": { "name": subj, "x": sbb.x1, "y": sbb.y1, "w": sbb.size()[0], "h": sbb.size()[1]},
                    "confidence": prob.item(),
                })
        return {name: res}

    def run(self, imgs):
        runner = self
        res = {}
        percent_step = 1
        prev_percent = -percent_step
        prev_img_cnt = 0
        prev_time = time.time()

        for i, (name, img) in enumerate(imgs.items()):
            if (i * 100) // len(imgs) >= prev_percent + percent_step:
                prev_percent = (i * 100) // len(imgs)
                cur_time = time.time()
                v = (i - prev_img_cnt) / (cur_time - prev_time)
                print(f"{prev_percent}% done.., at {v} imgs/s")
                prev_time = cur_time
                prev_img_cnt = i

            img_res = self.run_single_image(name, img)
            res[name] = img_res[name]

        with open("inference.json", "w") as f:
            json.dump(res, f, indent=4)


def combine_to_one_file(dir):
    res = {}
    for fname in os.listdir(dir):
        path = f"{dir}/{fname}"
        assert str.endswith(path, ".json")
        with open(path) as f:
            j = json.load(f)
            for k, v in j.items():
                res[k] = v
    return res

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    runner = Runner(args)

    if args.input is not None:
        f = open(args.input)
        runner.run(json.load(f))
        f.close()
    elif args.dir is not None:
        runner.run(combine_to_one_file(args.dir))
