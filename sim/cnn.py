"""
A small convolutional neural network for classifying conveyor camera frames.

Input: a frame downscaled to 96 x 72 (keeping the 4:3 aspect ratio).
Output: one of Red, Green, Blue, Other.

Four convolution blocks (3x3 conv, batch norm, ReLU, 2x2 max-pool), then
global average pooling and a linear layer. About 60k parameters, small enough
to run in a few milliseconds per frame on a laptop CPU.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

INPUT_W, INPUT_H = 96, 72


def preprocess(frames_bgr):
    """uint8 BGR frames [N,H,W,3] (any size) -> float tensor [N,3,72,96] in [0,1], RGB."""
    small = np.stack([cv2.resize(f, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
                      for f in frames_bgr])
    return to_tensor(small)


def to_tensor(small_bgr):
    """Already-resized uint8 BGR frames -> float tensor [N,3,H,W], RGB."""
    rgb = small_bgr[..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))


class SortingCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        return self.head(x.mean(dim=(2, 3)))


def train(model, x_train, y_train, x_val, y_val, epochs=10, batch=64, lr=1e-3, seed=0, log=print):
    """Train with Adam; keep the weights from the epoch with the best validation accuracy."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    best_acc, best_state = -1.0, None
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=gen)
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_acc = (predict(model, x_val) == y_val).mean()
        log(f"  epoch {epoch + 1:>2}: loss {total / len(order):.4f}, validation accuracy {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_acc


@torch.no_grad()
def predict(model, x, batch=256):
    model.eval()
    return np.concatenate([model(x[i:i + batch]).argmax(1).numpy() for i in range(0, len(x), batch)])
