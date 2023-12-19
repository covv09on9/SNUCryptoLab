from heaan_sdk.matrix import HEMatrix, HESubMatrix
from heaan_sdk.matrix.ops import mat_ops as mop


class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, X: HEMatrix):
        y = mop.sigmoid(X)
        self.y = y
        return y

    def backward(self, dout):
        dx = dout * self.y * (1 - self.y)
        return dx


class GeLU:
    def __init__(self):
        self.y = None

    def forward(self, X: HEMatrix):
        None

    def backward(self, dout):
        None


class Softmax:
    def __init__(self):
        self.y = None

    def forward(self, X: HEMatrix):
        None

    def backward(self, dout):
        None
