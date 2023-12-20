from heaan_sdk.matrix import HEMatrix, HESubMatrix
from heaan_sdk.matrix.ops import mat_ops as mop


class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, X:HEMatrix):
        y = mop.sigmoid(X)
        self.y = y
        return y

    def backward(self, dout):
        dx = dout * self.y * (1 - self.y)
        return dx


class GeLU:
    def __init__(self):
        self.y = None
        self.X = None

    def forward(self, X: HEMatrix):
        self.X = X
        y = self.X * mop.sigmoid(1.702 * self.X)
        self.y = y
        return y

    def backward(self, dout):
        dx = dout * (mop.sigmoid(self.X) + 1.702 * self.X * (self.y * (1 - self.y)))
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.y: HEMatrix = None
        self.loss = None
        self.t: HEMatrix = None

    @classmethod
    def cross_entropy_loss(cls, y: HEMatrix, t: HEMatrix):
        batch_size = y.num_rows
        return -mop.vertical_sum(mop.log10((y - t) + 1e-7)) / batch_size

    def forward(self, X: HEMatrix, t: HEMatrix):
        self.t = t
        self.y = mop.softmax(X)
        self.loss = self.cross_entropy_loss(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.y.num_rows
        dx = (self.y - self.t) / batch_size
        return dx
