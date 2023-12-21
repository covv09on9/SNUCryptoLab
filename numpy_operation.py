from abc import ABC, abstractmethod
from typing import Tuple, Optional, cast, List

from heaan_sdk.matrix import HEMatrix, HESubMatrix
from heaan_sdk.core.context import Context
from heaan_sdk.ml.linear_model import initializers, utils
from get_random_state import get_random_state
from heaan_sdk.matrix.ops import mat_ops as mop
from heaan_sdk.ml.linear_model.datasets import DataSet
from heaan_sdk.core.mod import heaan
import numpy as np
import pandas as pd
import numpy.typing as npt

NDArrayCplx: TypeAlias = npt.NDArray[Any]

class Operation(ABC):
    def __init__(self):
        pass 

    @abstractmethod
    def forward(self, x):
        pass 
    
    @abstractmethod
    def backward(self, dout):
        pass

class ParamOperation(Operation, ABC):
    def __init__(self):
        pass 
    @abstractmethod
    def forward(self, x):
        pass 
    
    @abstractmethod
    def backward(self, dout):
        pass

class ActivationOperation(Operation, ABC):
    def __init__(self):
            pass 
    
    @abstractmethod
    def forward(self, x):
        pass 
    
    @abstractmethod
    def backward(self, dout):
        pass

class Params(ParamOperation):
    def __init__(self,
                 input_dim:int, 
                 output_dim:int,
                 random_state:Optional[int] = 0,
                 )->None:
        
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.random_state = get_random_state(random_state)
        self.weight = self._init_theta()
        self.dweight = None 
        self.x = None
        
    @property
    def shape(self)->Tuple[int, int]:
        return (self.input_dim, self.output_dim)
    
    def _init_theta(self) -> NDArrayCplx:
        return np.random.rand(self.input_dim, self.output_dim)

    def forward(self, x:NDArrayCplx)->NDArrayCplx:
        self.x = x
        x = np.matmul(x, self.weight.T)
        return x
    
    def backward(self, dout:NDArrayCplx)->NDArrayCplx:
        dx = np.matmul(dout, self.weight.T) # dx = mop.mat_submat_diag_abt(dout, self.weight)
        self.dweight = np.matmul(dout.T, self.x) # self.dweight = mop.mat_diag_atb(dout, self.x)
        return dx
    
class Sigmoid(ActivationOperation):
    def __init__(self):
        self.y = None

    def forward(self, x:NDArrayCplx):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, dout:NDArrayCplx):
        tmp_dx:NDArrayCplx = self.y * (1 - self.y)
        #tmp_dx.bootstrap_if(3, force=True)
        #dout.bootstrap_if(3, force=True)
        dx = np.matmul(dout, tmp_dx)
        return dx
    
class GeLU(ActivationOperation):
    def __init__(self):
        self.y = None
        self.x = None

    def forward(self, x: NDArrayCplx):
        #x.bootstrap_if(3, force=True)
        self.x = x
        tmp_x = 1.702 * self.x
        #tmp_x.bootstrap_if(3, force=True)
        y = self.x * (1 / (1 + np.exp(-tmp_x)))
        #y.bootstrap_if(3, force=True)
        self.y = y
        return y

    def backward(self, dout):
        #self.x.bootstrap_if(3, force=True)
        #self.y.bootstrap_if(3, force=True)

        tmp_x:NDArrayCplx = 1.702 * self.X
        #tmp_x.bootstrap_if(3, force=True)

        tmp_y:NDArrayCplx = (self.y * (1 - self.y))
        #tmp_y.bootstrap_if(3, force=True)

        tmp_dout = 1 / (1 + np.exp(-tmp_x))
        tmp_dout2 = np.matul(tmp_x, tmp_y)  
        #tmp_dout2.bootstrap_if(3, force=True)

        dx:NDArrayCplx = np.matmul(dout, tmp_dout + tmp_dout2)
        #dx.bootstrap_if(3)
        return dx
    

class Layers:
    def __init__(self):
        self.layers:List[Operation] = []

    def forward(self, x:NDArrayCplx)->NDArrayCplx:
        if len(self.layers) > 0:
            for layer in self.layers :
                x = layer.forward(x)
            return x
        else:
            raise Exception("Layer should be more than one")
    
    def backward(self, dout:NDArrayCplx)->NDArrayCplx:
        if len(self.layers) > 0:
            for layer in reversed(self.layers):
                dout = layer.backward(dout)
            return dout 
        else:
            raise Exception("Layer should be more than one")

    def append(self, addOp:Operation):
        if len(self.layers) != 0 :
            lastLayer_outputDim = self.layers[-1].weight.shape[1]
        if isinstance(addOp, ParamOperation):
            assert addOp.weight.shape[0] == lastLayer_outputDim
        self.layers.append(addOp)

class MLP:
    def __init__(self, 
                 layers:List[Operation],
                 lr:float,
                 num_epoch:int,
                 batch_size:int,
                 ):
        self.epoch_state: int = 0
        self.step_state: int = 1
        self.lr = lr
        self.epoch = num_epoch
        self.batch_size = batch_size
        self.layers = layers
        self.grads = []
        
    def predict(self, x:NDArrayCplx):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def cross_entropy_loss(self, y, t):
        #batch_size = t.shape[0]
        #tmp = t * mop.loge(y + 1e-5)
        #tmp.bootstrap_if(3)
        #loss = mop.vertical_sum(tmp, direction=0, fill=True)
        #loss.num_cols = 1
        #loss.num_rows = 1
        #loss.bootstrap_if(3)
        #loss *= 1/batch_size
        delta = 1e-5
        batch_size = t.shape[0]
        loss = -np.sum(t*np.log(y+delta))/batch_size

        return loss
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = self.cross_entropy_loss(y, t)
        return loss
    
    def gradient(self, x, t):
        dout = self.loss(x, t)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        

    def fit(self, train:pd.DataFrame):
        total_list = list(range(len(train)))
        num_batch = min(int(ceil(self.batch_size / train.shape[0])), len(train))
        
        columns = train.columns.to_list()
        features = []
        for col in columns:
            if col != "target":
                features.append(col)
        X, t = train[features].to_numpy(), train["target"]

        for epoch in range(self.epoch):
            batch_set = [total_list[idx : idx + num_batch] for idx in range(0, len(train), num_batch)]
            batch_set = tqdm(batch_set, desc=f"Epoch {epoch + 1}")
            for batch_list in batch_set:
                bsz = utils.get_batch_size(X, batch_list)
                X_batch = []
                y_batch = []
                for i in batch_list:
                    X_batch.append(X[i])
                    y_batch.append(t[i])
                
                grads = self.gradient(X_batch, y_batch)
                for i in range(len(self.layers)):
                    self.layers[i].weight -= self.lr * self.layers[i].dweight

    #@property
    #def device(self) -> heaan.Device:
    #    return self.layers[0].weight.device


