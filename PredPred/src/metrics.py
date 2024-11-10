import torch
import os


class Metrics:
    def __init__(self, metrics_dir, pred_count):
        self.metrics_dir = metrics_dir
        if not os.path.exists(self.metrics_dir):
            os.mkdir(self.metrics_dir)

        self.pred_count = pred_count
        self.train_cm = torch.zeros(
            (self.pred_count, self.pred_count),
            dtype=torch.int64,
        )
        self.validation_cm = torch.zeros(
            (self.pred_count, self.pred_count),
            dtype=torch.int64,
        )
        self.train_loss = []
        self.grads = []
        self.validation_loss = []

    def training_batch(self, pred_ids, actual_ids, loss, grad_norm):
        self.train_loss.append(loss)
        self.grads.append(grad_norm)
        for i in range(0, len(pred_ids)):
            self.train_cm[pred_ids[i]][actual_ids[i]] += 1

    def validation_batch(self, pred_ids, actual_ids, loss):
        self.validation_loss.append(loss)
        for i in range(0, len(pred_ids)):
            self.validation_cm[pred_ids[i]][actual_ids[i]] += 1

    def end_epoch(self, epoch):
        torch.save(
            [
                self.grads,
                self.train_loss,
                self.validation_loss,
                self.train_cm,
                self.validation_cm,
            ],
            f"{self.metrics_dir}/metrics_{epoch}.dat",
        )
        self.grads = []
        self.train_loss = []
        self.validation_loss = []
        self.train_cm.fill_(0)
        self.validation_cm.fill_(0)
