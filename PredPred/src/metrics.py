import torch
import os


class Metrics:
    def __init__(self, metrics_dir):
        self.metrics_dir = metrics_dir
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir, exist_ok=True)

        self.train_pred = []
        self.train_y = []
        self.v_pred = []
        self.v_y = []
        self.train_loss = []
        self.grads = []
        self.v_loss = []

    def training_batch(self, pred, y, loss, grad_norm):
        self.train_loss.append(loss)
        self.grads.append(grad_norm)
        self.train_y.append(y.to("cpu"))
        self.train_pred.append(pred.to("cpu"))

    def validation_batch(self, pred, y, loss):
        self.v_loss.append(loss)
        self.v_y.append(y.to("cpu"))
        self.v_pred.append(pred.to("cpu"))

    def start_epoch(self, epoch):
        pass

    def end_epoch(self, epoch):
        torch.save(
            [
                self.train_pred,
                self.train_y,
                self.v_pred,
                self.v_y,
                self.grads,
                self.train_loss,
                self.v_loss,
            ],
            f"{self.metrics_dir}/metrics_{epoch:05d}.bin",
        )
        self.train_pred = []
        self.train_y = []
        self.v_pred = []
        self.v_y = []
        self.train_loss = []
        self.grads = []
        self.v_loss = []
