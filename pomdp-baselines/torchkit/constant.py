import torch.nn as nn
from ffm import DropInFFM

LSTM_name = "lstm"
GRU_name = "gru"
FFM_name = 'ffm'
RNNs = {
    LSTM_name: nn.LSTM,
    GRU_name: nn.GRU,
    FFM_name: DropInFFM
}
