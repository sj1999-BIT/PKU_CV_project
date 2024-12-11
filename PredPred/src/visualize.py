import matplotlib.pyplot as plt
import torch
import argparse
import sys
import os
from dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--metrics",
    help="Directory with metrics files",
)
parser.add_argument(
    "--output",
    help="Directory to save output images to",
    required=True,
)
parser.add_argument(
    "--dataset",
    help="Dataset file path",
)
parser.add_argument(
    "--confusion_matrix",
    help="Metrics to draw confusion matrix",
)

def draw_loss(args):
    t_loss = []
    v_loss = []
    t_acc = []
    v_acc = []
    for f in os.listdir(args.metrics):
        [tp, ty, vp, vy, _, tl, vl] = torch.load(f"{args.metrics}/{f}", weights_only=True)
        t_loss.append(torch.tensor(tl).mean())
        v_loss.append(torch.tensor(vl).mean())

        correct = 0
        total = 0
        for b in range(len(tp)):
            p = tp[b].argmax(1)
            correct += p.eq(ty[b].argmax(1)).sum().item()
            total += len(p)
        t_acc.append(correct / total)

        correct = 0
        total = 0
        for b in range(len(vp)):
            p = vp[b].argmax(1)
            correct += p.eq(vy[b].argmax(1)).sum().item()
            total += len(p)
        v_acc.append(correct / total)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Loss over epochs")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Cross Entropy Loss")
    ax.plot(t_loss, label="Train")
    ax.plot(v_loss, label="Validation")
    ax.legend()
    plt.savefig(f"{args.output}/loss.png")
    plt.close()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Accuracy over epochs")
    ax.set_xlabel("Epoch #")
    ax.plot(t_acc, label="Train")
    ax.plot(v_acc, label="Validation")
    ax.legend()
    plt.savefig(f"{args.output}/acc.png")
    plt.close()

    pass

def draw_dataset(args):
    ds = Dataset(args.dataset, "cpu")
    pred_counts = [0 for x in range(ds.output_size())]

    for y in ds.ys:
        y = y.argmax(0)
        pred_counts[y] += 1

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Predicate usage")
    pred_counts = torch.tensor(pred_counts)
    ax.tick_params(axis="x", labelrotation=90)
    ax.bar(ds.pred_names, pred_counts / pred_counts.sum())
    plt.savefig(f"{args.output}/predicates.png")
    plt.close()


def draw_confusion_matrix(args):
    ds = Dataset(args.dataset, "cpu")
    [tp, ty, vp, vy] = torch.load(args.confusion_matrix, weights_only=True)[:4]
    tcm = torch.zeros((ds.output_size(), ds.output_size()))
    for b in range(len(tp)):
        pred_ids = tp[b].argmax(1)
        actual_ids = ty[b].argmax(1)
        for i in range(len(pred_ids)):
            tcm[actual_ids[i]][pred_ids[i]] += 1

    vcm = torch.zeros((ds.output_size(), ds.output_size()))
    for b in range(len(vp)):
        pred_ids = vp[b].argmax(1)
        actual_ids = vy[b].argmax(1)
        for i in range(len(pred_ids)):
            vcm[actual_ids[i]][pred_ids[i]] += 1


    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(121)
    ax.set_title("Training confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    ticks = [i for i in range(len(ds.pred_names))]
    ax.set_yticks(ticks, ds.pred_names)
    ax.set_xticks(ticks, ds.pred_names)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis="x", labelrotation=90)
    ax.imshow(tcm / tcm.sum(dim=0))

    ax = fig.add_subplot(122)
    ax.set_title("Validation confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    ticks = [i for i in range(len(ds.pred_names))]
    ax.set_yticks(ticks, ds.pred_names)
    ax.set_xticks(ticks, ds.pred_names)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis="x", labelrotation=90)
    ax.imshow(vcm / vcm.sum(dim=0))

    plt.savefig(f"{args.output}/confusion.png")
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    if args.metrics is not None:
        draw_loss(args)

    if args.dataset is not None:
        draw_dataset(args)

    if args.dataset is not None\
            and args.confusion_matrix is not None:
        draw_confusion_matrix(args)
