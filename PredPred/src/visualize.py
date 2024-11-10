import matplotlib.pyplot as plt
import torch
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    help="Directory with metrics files",
    required=True,
)
parser.add_argument(
    "--output",
    help="Directory to save output images to",
    required=True,
)


def draw_basic_stats(args):
    epcoh_count = len(os.listdir(args.input))
    train_losses = [None for i in range(epcoh_count)]
    validation_losses = [None for i in range(epcoh_count)]
    train_accuracy = [None for i in range(epcoh_count)]
    validation_accuracy = [None for i in range(epcoh_count)]
    train_macrop = [None for i in range(epcoh_count)]
    train_macror = [None for i in range(epcoh_count)]
    train_microp = [None for i in range(epcoh_count)]
    train_micror = [None for i in range(epcoh_count)]
    validation_macrop = [None for i in range(epcoh_count)]
    validation_macror = [None for i in range(epcoh_count)]
    validation_microp = [None for i in range(epcoh_count)]
    validation_micror = [None for i in range(epcoh_count)]

    for f in os.listdir(args.input):
        f_name, _ = f.split('.')
        _, num = f_name.split('_')
        i = int(num)

        [_, tl, vl, tcm, vcm] = torch.load(
            f"{args.input}/{f}", weights_only=True)

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

        train_accuracy.append(t_acc)
        validation_accuracy.append(v_acc)
        train_macrop.append(t_macrop)
        train_macror.append(t_macror)
        train_microp.append(t_microp)
        train_micror.append(t_micror)
        validation_macrop.append(v_macrop)
        validation_macror.append(v_macror)
        validation_microp.append(v_microp)
        validation_micror.append(v_micror)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Loss over epochs")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Cross Entropy Loss")
    ax.plot(train_losses, label="Train")
    ax.plot(validation_losses, label="Validation")
    ax.legend()
    plt.savefig(f"{args.output}/loss.png")

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


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    draw_basic_stats(args)
