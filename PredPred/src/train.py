from dataset import DataSet
import logging
import argparse
import sys
import model
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from metrics import Metrics

parser = argparse.ArgumentParser(
    prog="PredPred",
    description="Model for predicting predicates",
)
parser.add_argument(
    "--input",
    help="dataset file path",
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
    "--layer_1",
    help="Dimenision of layer 1",
    default=5000,
    type=int,
)
parser.add_argument(
    "--layer_2",
    help="Dimenision of layer 2",
    default=10000,
    type=int,
)
parser.add_argument(
    "--layer_3",
    help="Dimenision of layer 3",
    default=5000,
    type=int,
)
parser.add_argument(
    "--seed",
    help="Random seed for splitting the dataset",
    default=0,
    type=int,
)
parser.add_argument(
    "--metrics_dir",
    help="Directory for saving the metrics",
    default="metrics",
)
parser.add_argument(
    "--log_file",
    help="File to save logs to",
    default="out.log",
)


class Trainer:
    def __init__(self, args):
        self.args = args

        self.ds = DataSet(args.input, args.split, args.seed)
        self.train_set = DataLoader(
            self.ds.train_set(),
            batch_size=args.batch_size,
            shuffle=True,
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
        self.m = model.Model(
            110,
            self.args.layer_1, self.args.layer_2, self.args.layer_3,
            self.ds.pred_count,
        ).to(args.device)
        logging.info("Creating the model done")

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.m.parameters(),
            lr=args.learning_rate
        )

        self.metrics = Metrics(self.args.metrics_dir, self.ds.pred_count)

        if not os.path.exists(self.args.model_dir):
            os.mkdir(self.args.model_dir)

    def train(self):
        for i in range(0, self.args.epoch_count):
            logging.info(f"==================Epoch {i}=================")
            self.train_epoch(i)
            self.validate_epoch(i)

            model_name = f"{self.args.model_dir}/model_{i}.pth"
            torch.save(self.m.state_dict(), model_name)
            self.metrics.end_epoch(i)

    def validate_epoch(self, epoch_idx):
        logging.info(f"Validation for epoch {epoch_idx}")
        self.m.eval()
        total_loss = 0.0
        correct = 0

        with torch.no_grad():
            for (x, i) in self.validation_set:
                x = x.to(self.args.device)
                y = torch.zeros(len(x), self.ds.pred_count).\
                    scatter_(1, i.unsqueeze(1), 1).\
                    to(self.args.device)
                pred = self.m(x)
                pred_ids = pred.argmax(1).to("cpu")
                correct += pred_ids.eq(i.to("cpu")).sum().item()
                loss_amt = self.loss_fn(pred, y).item()
                total_loss += loss_amt

                self.metrics.validation_batch(pred_ids, i, loss_amt)

        avg_loss = total_loss / len(self.validation_set)
        accuracy = correct / len(self.validation_set.dataset)
        logging.info(f"Validation done for epoch {epoch_idx}")
        logging.info(f"Loss: {avg_loss}")
        logging.info(f"Accuracy: {accuracy}")

    def train_epoch(self, epoch_idx):
        assert torch.is_grad_enabled()
        logging.info(f"Training epoch {epoch_idx}")
        self.m.train()
        size = len(self.train_set)
        correct = 0
        total = 0
        batches = 0
        total_grad_norm = 0.0
        total_loss_amt = 0.0

        for batch, (x, i) in enumerate(self.train_set):
            self.optimizer.zero_grad()

            x = x.to(self.args.device)
            y = torch.zeros(len(x), self.ds.pred_count).\
                scatter_(1, i.unsqueeze(1), 1).\
                to(self.args.device)
            pred = self.m(x)
            pred_ids = pred.argmax(1).to("cpu")
            correct += pred_ids.eq(i).sum().item()
            total += len(x)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            loss_amt = loss.item()
            total_loss_amt += loss_amt

            grad_norm = self.grad_norm()
            total_grad_norm += grad_norm
            self.metrics.training_batch(pred_ids, i, loss_amt, grad_norm)
            batches += 1

            if batch % 1000 == 0 or batch + 1 == size:
                logging.info(f"Batch {batch}/{size}:")
                logging.info(f"=======> AvgLoss: {total_loss_amt / batches}")
                logging.info(f"=======> Accuracy: {correct / total}")
                logging.info(f"=======> AvgGrad: {total_grad_norm / batches}")
                batches = 0
                total_grad_norm = 0.0
                total_loss_amt = 0.0
                correct = 0
                total = 0

        logging.info(f"Finished training epoch {epoch_idx}")

    def grad_norm(self):
        total = 0.0
        for p in self.m.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5


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
    logging.info(f"Args: {args}")
    logging.info("Started")
    t = Trainer(args)
    t.train()
    logging.info("Finished")
