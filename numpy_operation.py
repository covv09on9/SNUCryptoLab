from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, cast, List

import numpy as np
import pandas as pd
import numpy.typing as npt
from tqdm import tqdm
from math import ceil

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
                 )->None:
        
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.weight = self._init_theta()
        self.dweight = None 
        self.x = None
        
    @property
    def shape(self)->Tuple[int, int]:
        return self.weight.shape
    
    def _init_theta(self) -> np.ndarray:
        return np.random.rand(self.output_dim, self.input_dim)

    def forward(self, x:np.ndarray)->np.ndarray:
        x = np.array(x)
        self.x = x
        x = np.matmul(x, self.weight.T)
        return x
    
    def backward(self, dout:np.ndarray)->np.ndarray:
        dout = np.array(dout)
        dx = np.matmul(dout, self.weight) # dx = mop.mat_submat_diag_abt(dout, self.weight)
        self.dweight = np.matmul(self.x.T, dout) # self.dweight = mop.mat_diag_atb(dout, self.x)
        return dx
    
class Sigmoid(ActivationOperation):
    def __init__(self):
        self.y = None

    def forward(self, x:np.ndarray):
        x = np.array(x)
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, dout:np.ndarray):
        dout = np.array(dout)
        tmp_dx:np.ndarray = self.y * (1 - self.y)
        #tmp_dx.bootstrap_if(3, force=True)
        #dout.bootstrap_if(3, force=True)
        dx = dout * tmp_dx
        return dx
    
class GeLU(ActivationOperation):
    def __init__(self):
        self.y = None
        self.x = None

    def forward(self, x: np.ndarray):
        x = np.array(x)
        #x.bootstrap_if(3, force=True)
        self.x = x
        tmp_x = 1.702 * self.x
        #tmp_x.bootstrap_if(3, force=True)
        y = self.x * (1 / (1 + np.exp(-tmp_x)))
        #y.bootstrap_if(3, force=True)
        self.y = y
        return y

    def backward(self, dout):
        dout = np.array(dout)
        tmp_x:np.ndarray = 1.702 * self.x

        tmp_y:np.ndarray = (self.y * (1 - self.y))

        tmp_dout = 1 / (1 + np.exp(-tmp_x))
        tmp_dout2 = tmp_x * tmp_y  
        #tmp_dout2.bootstrap_if(3, force=True)

        dx:np.ndarray = dout * (tmp_dout + tmp_dout2)
        #dx.bootstrap_if(3)
        return dx
    
class Layers:
    def __init__(self):
        self.layers:List[Operation] = []

    def forward(self, x:np.ndarray)->np.ndarray:
        if len(self.layers) > 0:
            for layer in self.layers :
                x = layer.forward(x)
            return x
        else:
            raise Exception("Layer should be more than one")
    
    def backward(self, dout:np.ndarray)->np.ndarray:
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
        
    def predict(self, x:np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def cross_entropy_loss(self, y, t):
        delta = 1e-5
        batch_size = len(t)
        loss = -np.sum(t*np.log(y+delta))/batch_size

        return loss
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = self.cross_entropy_loss(y, t)
        loss = np.array([loss])
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
                X_batch = []
                y_batch = []
                for i in batch_list:
                    X_batch.append(X[i])
                    y_batch.append(t[i])
                grads = self.gradient(X_batch, y_batch)
                for i, layer in enumerate(self.layers):
                    if isinstance(layer, ParamOperation):
                        self.layers[i].weight -= self.lr * self.layers[i].dweight.T