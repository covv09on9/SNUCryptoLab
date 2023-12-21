from abc import ABC, abstractmethod
from typing import Tuple, Optional, cast, List

from heaan_sdk.core import HEObject
from heaan_sdk.matrix import HEMatrix, HESubMatrix
from heaan_sdk.core.context import Context
from heaan_sdk.ml.linear_model import initializers, utils
from heaan_sdk.util import as_shape, get_random_state
from heaan_sdk.matrix.ops import mat_ops as mop
from heaan_sdk.ml.linear_model.datasets import DataSet
from typing_extensions import Self
from heaan_sdk.core.mod import heaan
from math import ceil
from tqdm import tqdm 

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
                 context:Context, 
                 unit_shape:Tuple[int, ...],
                 input_dim:int, 
                 output_dim:int,
                 random_state:Optional[int] = 0,
                 initializer:str = "kaiming_he",
                 )->None:
        self.context = context
        self.unit_shape = unit_shape
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.random_state = get_random_state(random_state)
        self.weight = self._init_theta(initializer)
        self.dweight = None 
        self.x = None
        
    @property
    def shape(self)->Tuple[int, int]:
        return (self.output_dim, self.input_dim)
    
    def _init_theta(self, initializer: str = "kaiming_he") -> HESubMatrix:
        return cast(
            HESubMatrix,
            getattr(initializers, initializer)(self.context, self.shape, self.unit_shape, self.random_state),
        )

    def forward(self, x:HEMatrix)->HEMatrix:
        self.x = x
        x = mop.mat_submat_diag_abt(x, self.weight)
        return x 
    
    def backward(self, dout:HEMatrix)->HEMatrix:
        dx = mop.mat_submat_diag_abt(dout, self.weight)
        self.dweight = mop.mat_diag_atb(dout, self.x)
        return dx
    
class Sigmoid(ActivationOperation):
    def __init__(self):
        self.y = None

    def forward(self, x:HEMatrix):
        y = mop.sigmoid(x)
        self.y = y
        return y

    def backward(self, dout:HEMatrix):
        tmp_dx:HEMatrix = self.y * (1 - self.y)
        tmp_dx.bootstrap_if(3, force=True)
        dout.bootstrap_if(3, force=True)
        dx = dout * tmp_dx
        return dx
    
class GeLU(ActivationOperation):
    def __init__(self):
        self.y = None
        self.x = None

    def forward(self, x: HEMatrix):
        x.bootstrap_if(3, force=True)
        self.x = x
        tmp_x = 1.702 * self.x
        tmp_x.bootstrap_if(3, force=True)
        y = self.x * mop.sigmoid(tmp_x)
        y.bootstrap_if(3, force=True)
        self.y = y
        return y

    def backward(self, dout):
        self.x.bootstrap_if(3, force=True)
        self.y.bootstrap_if(3, force=True)

        tmp_x:HEMatrix = 1.702 * self.X
        tmp_x.bootstrap_if(3, force=True)

        tmp_y:HEMatrix = (self.y * (1 - self.y))
        tmp_y.bootstrap_if(3, force=True)

        tmp_dout = mop.sigmoid(tmp_x)
        tmp_dout2 = tmp_x * tmp_y 
        tmp_dout2.bootstrap_if(3, force=True)

        dx:HEMatrix = dout * (tmp_dout + tmp_dout2)
        dx.bootstrap_if(3)
        return dx
    

class Layers:
    def __init__(self):
        self.layers:List[Operation] = []

    def forward(self, x:HEMatrix)->HEMatrix:
        if len(self.layers) > 0:
            for layer in self.layers :
                x = layer.forward(x)
            return x
        else:
            raise Exception("Layer should be more than one")
    
    def backward(self, dout:HEMatrix)->HEMatrix:
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
        
    def predict(self, x:HEMatrix):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def cross_entropy_loss(self, y, t):
        batch_size = t.shape[0]
        tmp = t * mop.loge(y + 1e-5)
        tmp.bootstrap_if(3)
        loss = mop.vertical_sum(tmp, direction=0, fill=True)
        loss.num_cols = 1
        loss.num_rows = 1
        loss.bootstrap_if(3)
        loss *= 1/batch_size
        return (-1 * loss) 
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = self.cross_entropy_loss(y, t)
        return loss
    
    def gradient(self, x, t):
        dout = self.loss(x, t)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        for layer in self.layers :
            
    def fit(self, train:DataSet):
        total_list = list(range(len(train)))
        num_batch = min(int(ceil(self.batch_size / train.unit_shape[0])), len(train))
        
        X, t = train.feature, train.target

        for epoch in range(self.epoch):
            batch_set = [total_list[idx : idx + num_batch] for idx in range(0, len(train), num_batch)]
            batch_set = tqdm(batch_set, desc=f"Epoch {epoch + 1}")
            for batch_list in batch_set:
                batch_list = self._reorder_batch(train, batch_list)
                bsz = utils.get_batch_size(X, batch_list)
                X_batch = HEMatrix(self.context, shape=(bsz, X.shape[1]), encrypted=train.encrypted)
                y_batch = HEMatrix(self.context, shape=(bsz, t.shape[1]), encrypted=train.encrypted)
                for i in batch_list:
                    X_batch.objects.append(X[i].deepcopy())
                    y_batch.objects.append(t[i].deepcopy())
                X_batch.to(self.device)
                y_batch.to(self.device)
                self.gradient(X_batch, y_batch)
                for layer in self.layers:
                    if isinstance(layer, ParamOperation):
                        layer.weight -= self.lr * self.layer.dweight

    @property
    def device(self) -> heaan.Device:
        return self.layers[0].weight.device

    def _reorder_batch(self, data_set: DataSet, batch_list: List[int]) -> List[int]:
        """Reorder batch index so that the remaining submatrix became the last submatrix in a batch.

        Args:
            data_set (DataSet): DataSet.
            batch_list (List[int]): List of batch index.

        Returns:
            List[int]: Reordered list of batch index.
        """
        num_batch = len(batch_list)
        assert data_set.target is not None
        for i in range(num_batch - 1):
            if data_set.target[batch_list[i]].shape[0] < data_set.unit_shape[0]:
                batch_list[i], batch_list[-1] = batch_list[-1], batch_list[i]
                break
        return batch_list