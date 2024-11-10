import torch
import argparse
import sys
from dataset import DataSet
from dataset import Bounds
from glove import Glove
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help="Path to a model file to perform inference with",
    required=True,
)
parser.add_argument(
    "--layer_1",
    help="Size of layer 1",
    required=True,
    type=int,
)
parser.add_argument(
    "--layer_2",
    help="Size of layer 2",
    required=True,
    type=int,
)
parser.add_argument(
    "--layer_3",
    help="Size of layer 3",
    required=True,
    type=int,
)
parser.add_argument(
    "--glove",
    help="Path to glove.6B.50d.txt",
    required=True,
)
parser.add_argument(
    "--dataset",
    help="Path to dataset.bin",
    required=True,
)
parser.add_argument(
    "--device",
    help="Device for running model on",
    default="cpu"
)


class Runner:
    def __init__(self, args):
        self.args = args
        self.glove = Glove(args.glove)
        self.ds = DataSet(args.dataset, 10, 0)
        self.m = Model(108, args.layer_1, args.layer_2,
                       args.layer_3, self.ds.pred_count)
        self.m.load_state_dict(torch.load(
            args.model, weights_only=True, map_location=args.device))
        self.m.to(args.device)
        self.m.eval()

    def eval(self, obj_name, obj_bb, subj_name, subj_bb, n=10):
        o, s = obj_bb.normalize(subj_bb)
        vec = torch.concat((
            self.glove.get(obj_name),
            torch.tensor([o.x1, o.y1, o.size()[0], o.size()[1]]),
            self.glove.get(subj_name),
            torch.tensor([s.x1, o.y1, s.size()[0], s.size()[1]]),
        )).to(self.args.device)
        vec = vec.reshape((1, len(vec)))
        with torch.no_grad():
            pred = self.m(vec)
        vals, inds = torch.topk(pred, n)
        preds = []
        for i in inds[0]:
            preds.append(self.ds.get_predicate(i))

        return vals[0], preds


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    runner = Runner(args)

    print(runner.eval("window", Bounds.from_corner_size(602, 4, 173, 148),
          "building", Bounds.from_corner_size(1, 2, 536, 218)))
