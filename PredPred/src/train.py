from dataset import DataSet
import logging
import argparse
import sys
import model
import torch
from torch import nn
from torch.utils.data import DataLoader

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
    "--layer_1",
    help="Dimenision of layer 1",
    default=1000,
    type=int,
)
parser.add_argument(
    "--layer_2",
    help="Dimenision of layer 2",
    default=1000,
    type=int,
)


class Trainer:
    def __init__(self, args):
        self.args = args

        self.ds = DataSet(args.input, args.split)
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
        self.m = model.Model(
            108, self.ds.pred_count, self.args.layer_1, self.args.layer_2
        ).to(args.device)
        logging.info("Creating the model done")

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.m.parameters(),
            lr=args.learning_rate
        )

        self.train_loss = []
        self.validation_loss = []
        self.validation_accuracy = []

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
        correct = 0

        with torch.no_grad():
            for (x, i) in self.validation_set:
                x = x.to(self.args.device)
                y = torch.zeros(len(x), self.ds.pred_count).\
                    scatter_(1, i.unsqueeze(1), 1).\
                    to(self.args.device)
                pred = self.m(x)
                if pred.argmax(1) == i:
                    correct += 1
                total_loss += self.loss_fn(pred, y).item()

        avg_loss = total_loss / len(self.validation_set)
        accuracy = correct / len(self.validation_set.dataset)
        logging.info(f"Loss: {avg_loss}")
        logging.info(f"Accuracy: {accuracy}")
        self.validation_loss.append(avg_loss)
        self.validation_accuracy.append(accuracy)

        logging.info(f"Validation done for epoch {epoch_idx}")

    def train_epoch(self, epoch_idx):
        assert torch.is_grad_enabled()
        logging.info(f"Training epoch {epoch_idx}")
        self.m.train()
        size = len(self.train_set)

        for batch, (x, i) in enumerate(self.train_set):
            x = x.to(self.args.device)
            y = torch.zeros(len(x), self.ds.pred_count).\
                scatter_(1, i.unsqueeze(1), 1).\
                to(self.args.device)
            pred = self.m(x)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 1000 == 0:
                loss_amt = loss.item()
                logging.info(f"[{batch}/{size}] Loss: {loss_amt}")
                self.train_loss.append(loss_amt)

        logging.info(f"Finished training epoch {epoch_idx}")


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    logging.basicConfig(level=logging._nameToLevel[args.log])
    logging.info(f"Args: {args}")
    logging.info("Started")
    t = Trainer(args)
    t.train()
    logging.info("Finished")
