import matplotlib.pyplot as plt
import torch
import argparse
import sys
import os
from dataset import DataSet

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
    help="Metrics file to draw confusion matrix from",
)


def draw_basic_stats(args):
    epoch_count = len(os.listdir(args.metrics))
    train_losses = [None for i in range(epoch_count)]
    validation_losses = [None for i in range(epoch_count)]
    train_accuracy = [None for i in range(epoch_count)]
    validation_accuracy = [None for i in range(epoch_count)]
    train_macrop = [None for i in range(epoch_count)]
    train_macror = [None for i in range(epoch_count)]
    train_microp = [None for i in range(epoch_count)]
    train_micror = [None for i in range(epoch_count)]
    validation_macrop = [None for i in range(epoch_count)]
    validation_macror = [None for i in range(epoch_count)]
    validation_microp = [None for i in range(epoch_count)]
    validation_micror = [None for i in range(epoch_count)]

    for f in os.listdir(args.metrics):
        f_name, _ = f.split('.')
        _, num = f_name.split('_')
        i = int(num)

        [_, tl, vl, tcm, vcm] = torch.load(
            f"{args.metrics}/{f}", weights_only=True)

        tl = torch.tensor(tl)
        vl = torch.tensor(vl)

        train_losses[i] = torch.mean(tl)
        validation_losses[i] = torch.mean(vl)

        t_tpi = torch.diagonal(tcm)
        t_fpi = tcm.sum(dim=1) - t_tpi
        t_fni = tcm.sum(dim=0) - t_tpi
        t_pi = t_tpi / (t_tpi + t_fpi)
        t_ri = t_tpi / (t_tpi + t_fni)
        t_macrop = t_pi.nanmean()
        t_macror = t_ri.nanmean()
        t_microp = t_tpi.sum() / (t_tpi.sum() + t_fpi.sum())
        t_micror = t_tpi.sum() / (t_tpi.sum() + t_fni.sum())

        v_tpi = torch.diagonal(vcm)
        v_fpi = vcm.sum(dim=1) - v_tpi
        v_fni = vcm.sum(dim=0) - v_tpi
        v_pi = v_tpi / (v_tpi + v_fpi)
        v_ri = v_tpi / (v_tpi + v_fni)
        v_macrop = v_pi.nanmean()
        v_macror = v_ri.nanmean()
        v_microp = v_tpi.sum() / (v_tpi.sum() + v_fpi.sum())
        v_micror = v_tpi.sum() / (v_tpi.sum() + v_fni.sum())

        t_acc = t_tpi.sum() / tcm.sum()
        v_acc = v_tpi.sum() / vcm.sum()

        train_accuracy[i] = t_acc
        validation_accuracy[i] = v_acc
        train_macrop[i] = t_macrop
        train_macror[i] = t_macror
        train_microp[i] = t_microp
        train_micror[i] = t_micror
        validation_macrop[i] = v_macrop
        validation_macror[i] = v_macror
        validation_microp[i] = v_microp
        validation_micror[i] = v_micror

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Loss over epochs")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Cross Entropy Loss")
    ax.plot(train_losses, label="Train")
    ax.plot(validation_losses, label="Validation")
    ax.legend()
    plt.savefig(f"{args.output}/loss.png")
    plt.close()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Accuracy over epochs")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.plot(train_accuracy, label="Train")
    ax.plot(validation_accuracy, label="Validation")
    ax.legend()
    plt.savefig(f"{args.output}/accuracy.png")
    plt.close()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Precision and Recall")
    ax.set_xlabel("Epoch #")
    ax.set_ylim(0, 1)
    ax.plot(train_macrop, label="T. MacroPrecision")
    ax.plot(train_macror, label="T. MacroRecall")
    ax.plot(train_microp, label="T. MicroPrecision")
    ax.plot(train_micror, label="T. MicroRecall")
    ax.plot(validation_macrop, label="V. MacroPrecision")
    ax.plot(validation_macror, label="V. MacroRecall")
    ax.plot(validation_microp, label="V. MicroPrecision")
    ax.plot(validation_micror, label="V. MicroRecall")
    ax.legend()
    plt.savefig(f"{args.output}/precision.png")
    plt.close()


def draw_dataset(args):
    ds = DataSet(args.dataset, 10, 0)
    pred_counts = [0 for x in range(ds.pred_count)]

    for y in ds.ys:
        pred_counts[y] += 1

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Predicate usage")
    pred_counts = torch.tensor(pred_counts)
    ax.tick_params(axis="x", labelrotation=90)
    ax.bar(ds.id_to_pred, pred_counts / pred_counts.sum())
    plt.savefig(f"{args.output}/predicates.png")
    plt.close()


def draw_confusion_matrix(args):
    [_, _, _, tcm, vcm] = torch.load(args.confusion_matrix, weights_only=True)
    ds = DataSet(args.dataset, 10, 0)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121)
    ax.set_title("Training confusion matrix")
    ax.set_ylabel("Predicted")
    ax.set_xlabel("Expected")
    ticks = [i for i in range(len(ds.id_to_pred))]
    ax.set_yticks(ticks, ds.id_to_pred)
    ax.set_xticks(ticks, ds.id_to_pred)
    ax.tick_params(axis="x", labelrotation=90)
    ax.imshow(tcm / tcm.sum(dim=0))

    ax = fig.add_subplot(122)
    ax.set_title("Validation confusion matrix")
    ax.set_ylabel("Predicted")
    ax.set_xlabel("Expected")
    ax.set_yticks(ticks, ds.id_to_pred)
    ax.set_xticks(ticks, ds.id_to_pred)
    ax.tick_params(axis="x", labelrotation=90)
    ax.imshow(vcm / vcm.sum(dim=0))

    plt.savefig(f"{args.output}/confusion.png")
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if args.metrics is not None:
        draw_basic_stats(args)

    if args.dataset is not None:
        draw_dataset(args)

    if args.dataset is not None\
            and args.confusion_matrix is not None:
        draw_confusion_matrix(args)
