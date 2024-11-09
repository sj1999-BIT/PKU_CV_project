from dataset import DataSet
from glove import Glove
import logging
import argparse
import sys
import model
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    prog="PredPred",
    description="Model for predicting predicates",
)
parser.add_argument(
    "--input",
    help="relationships.json file path",
    required=True
)
parser.add_argument("--log", help="log level", default="INFO")
parser.add_argument(
    "--split",
    help="% of data to be used for validation",
    type=float,
    default=20,
)
parser.add_argument("--device", help="Device to train on", default="cpu")
parser.add_argument(
    "--learning_rate",
    help="Learning rate of the model",
    default=1e-3,
    type=float
)
parser.add_argument(
    "--epoch_count",
    help="Number of epochs to train",
    default=10,
    type=int,
)
parser.add_argument(
    "--batch_size",
    help="Batch size for training",
    default=64,
    type=int,
)
parser.add_argument(
    "--model_dir",
    help="Model backups directory",
    default="./models",
)
parser.add_argument(
    "--glove",
    help="File path to glove.6B.50d.txt",
    required=True,
)


class Trainer:
    def __init__(self, args):
        self.args = args

        self.glove = Glove(args.glove)
        self.ds = DataSet(self.glove, args.input, args.split)
        self.train_set = DataLoader(
            self.ds.train_set(),
            batch_size=args.batch_size
        )
        self.test_set = DataLoader(
            self.ds.test_set(),
            batch_size=args.batch_size
        )
        self.validation_set = DataLoader(
            self.ds.validation_set(),
            batch_size=args.batch_size
        )

        logging.info("Reading the dataset done")
        logging.info(f"Creating the model on device '{args.device}'")
        self.m = model.Model(108, self.ds.pred_count).to(args.device)
        logging.info("Creating the model done")

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.m.parameters(),
            lr=args.learning_rate
        )

        self.train_loss = []
        self.train_loss_x = []
        self.validation_loss = []
        self.validation_loss_x = []

    def plot_loss(self):
        fig = plt.figure(figsize=(20, 10))
        self.ax_loss = fig.add_subplot(111)
        self.ax_loss.set_xlabel("Epochs")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_title("Loss over time")
        self.ax_loss.set_xlim(0, self.args.epoch_count)
        self.ax_loss.set_ylim(0, 20)
        self.ax_loss.plot(
            self.train_loss_x,
            self.train_loss,
            label="Train loss",
            color="blue",
        )
        self.ax_loss.plot(
            self.validation_loss_x,
            self.validation_loss,
            label="Validation loss",
            color="red",
        )
        plt.legend()
        plt.show()

    def train(self):
        for i in range(0, self.args.epoch_count):
            logging.info(f"Staring epoch {i}")
            self.train_epoch(i)
            model_name = f"{self.args.model_dir}/model_{i}.pth"
            logging.info(f"Saving model as '{model_name}'")
            torch.save(self.m.state_dict(), model_name)
            logging.info("Model saved")
            self.validate_epoch(i)

    def validate_epoch(self, epoch_idx):
        logging.info(f"Validation for epoch {epoch_idx}")
        self.m.eval()
        total_loss = 0.0

        with torch.no_grad():
            for (x, y) in self.validation_set:
                x = x.to(self.args.device)
                y = torch.zeros(len(x), self.ds.pred_count).\
                    scatter_(1, y.unsqueeze(1), 1).\
                    to(self.args.device)
                pred = self.m(x)
                total_loss += self.loss_fn(pred, y).item()

        avg_loss = total_loss / len(self.validation_set)
        logging.info(f"Average loss: {avg_loss}")
        self.validation_loss.append(avg_loss)
        self.validation_loss_x.append(epoch_idx)

        logging.info(f"Validation done for epoch {epoch_idx}")

    def train_epoch(self, epoch_idx):
        logging.info(f"Training epoch {epoch_idx}")
        self.m.train()
        size = len(self.train_set)

        for batch, (x, y) in enumerate(self.train_set):
            x = x.to(self.args.device)
            y = torch.zeros(len(x), self.ds.pred_count).\
                scatter_(1, y.unsqueeze(1), 1).\
                to(self.args.device)
            pred = self.m(x)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 50 == 0:
                loss_amt = loss.item()
                logging.info(f"[{batch}/{size}] loss  = {loss_amt}")
                self.train_loss.append(loss_amt)
                self.train_loss_x.append(epoch_idx + batch * len(x) / size)

        logging.info(f"Finished training epoch {epoch_idx}")


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    logging.basicConfig(level=logging._nameToLevel[args.log])
    logging.info(f"Args: {args}")
    logging.info("Started")
    t = Trainer(args)
    t.train()
    logging.info("Finished")
    t.plot_loss()
