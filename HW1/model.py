from typing import Dict

import torch
from torch.nn import Embedding

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.RNN = torch.nn.LSTM(
            input_size = 300, hidden_size = hidden_size, num_layers = num_layers, 
            batch_first = True, dropout = dropout, bidirectional = bidirectional
        )
        D = 2 if bidirectional == True else 1
        self.fc_layer = torch.nn.Linear(hidden_size * D, hidden_size)
        self.normalization = torch.nn.BatchNorm1d(hidden_size)
        self.activation = torch.nn.Tanh()
        self.classifier = torch.nn.Linear(hidden_size, num_class)
        # self.linear_U = torch.nn.Linear(300, hidden_size)
        # self.linear_V = torch.nn.Linear(hidden_size, num_class)
        # self.linear_W = torch.nn.Linear(hidden_size, hidden_size)
        # self.activation = torch.nn.Tanh()

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embed_vector = self.embed(batch) # shape (batch_size, pad_len) -> (batch_size, pad_len, 300)
        RNN_output, (hn, cn) = self.RNN(embed_vector)
        fc_output = self.fc_layer(RNN_output[:, -1, :])
        z = self.activation(self.normalization(fc_output))
        y = self.classifier(z)

        # pad_len = len(batch[0])
        # hidden = torch.zeros(1, self.hidden_size)
        # for i in range(pad_len):
        #     Ux = self.linear_U(embed_vector[:,i,:])
        #     hidden = self.activation(self.linear_W(hidden) + Ux)
        # y = self.linear_V(hidden) # shape (batch_size, num_class)
        return y


class TagClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(TagClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.RNN = torch.nn.LSTM(
            input_size = 300, hidden_size = hidden_size, num_layers = num_layers, 
            batch_first = True, dropout = dropout, bidirectional = bidirectional
        )
        D = 2 if bidirectional == True else 1
        self.classifier = torch.nn.Linear(hidden_size, num_class)
        self.fc_layer = torch.nn.Linear(hidden_size * D, hidden_size)
        self.activation = torch.nn.Tanh()
        self.normalization = torch.nn.BatchNorm1d(hidden_size)
        self.layernorm = torch.nn.LayerNorm(embeddings.shape[-1])

        # self.linear_U = torch.nn.Linear(300, hidden_size)
        # self.linear_V = torch.nn.Linear(hidden_size, num_class)
        # self.linear_W = torch.nn.Linear(hidden_size, hidden_size)
        # self.activation = torch.nn.Tanh()
        # self.softmax = torch.nn.Softmax(dim = 1)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        pad_len = len(batch[0])
        embed_vector = self.embed(batch) # shape (batch_size, pad_len) -> (batch_size, pad_len, 300)
        normalized_embed = self.layernorm(embed_vector)
        RNN_output, (hn, cn) = self.RNN(embed_vector) #  shape (batch_size, pad_len, 300) -> (batch_size, pad_len, hidden_size * D)
        fc_output = self.fc_layer(RNN_output)
        normalized_output = self.normalization(fc_output.transpose(1, 2)).transpose(1, 2)
        z = self.activation(normalized_output)
        y = self.classifier(z)

        # y = self.classifier(RNN_output) # shape (batch_size, pad_len, hidden_size * D) -> (batch_size, pad_len, num_class)
        # pad_len = len(batch[0])
        # hidden = torch.zeros(1, self.hidden_size)
        # for i in range(pad_len):
        #     Ux = self.linear_U(embed_vector[:,i,:])
        #     hidden = self.activation(self.linear_W(hidden) + Ux)
        # y = self.softmax(self.linear_V(hidden)) # shape (batch_size, num_class)
        return y

class GRUSeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(GRUSeqClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.RNN = torch.nn.GRU(
            input_size = 300, hidden_size = hidden_size, num_layers = num_layers, 
            batch_first = True, dropout = dropout, bidirectional = bidirectional
        )
        D = 2 if bidirectional == True else 1
        self.fc_layer = torch.nn.Linear(hidden_size * D, hidden_size)
        self.normalization = torch.nn.BatchNorm1d(hidden_size)
        self.activation = torch.nn.Tanh()
        self.classifier = torch.nn.Linear(hidden_size, num_class)
        # self.linear_U = torch.nn.Linear(300, hidden_size)
        # self.linear_V = torch.nn.Linear(hidden_size, num_class)
        # self.linear_W = torch.nn.Linear(hidden_size, hidden_size)
        # self.activation = torch.nn.Tanh()

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embed_vector = self.embed(batch) # shape (batch_size, pad_len) -> (batch_size, pad_len, 300)
        RNN_output, hn = self.RNN(embed_vector)
        fc_output = self.fc_layer(RNN_output[:, -1, :])
        z = self.activation(self.normalization(fc_output))
        y = self.classifier(z)

        # pad_len = len(batch[0])
        # hidden = torch.zeros(1, self.hidden_size)
        # for i in range(pad_len):
        #     Ux = self.linear_U(embed_vector[:,i,:])
        #     hidden = self.activation(self.linear_W(hidden) + Ux)
        # y = self.linear_V(hidden) # shape (batch_size, num_class)
        return y

class GRUTagClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(GRUTagClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.RNN = torch.nn.GRU(
            input_size = 300, hidden_size = hidden_size, num_layers = num_layers, 
            batch_first = True, dropout = dropout, bidirectional = bidirectional
        )
        D = 2 if bidirectional == True else 1
        self.classifier = torch.nn.Linear(hidden_size, num_class)
        self.fc_layer = torch.nn.Linear(hidden_size * D, hidden_size)
        self.activation = torch.nn.Tanh()
        self.normalization = torch.nn.BatchNorm1d(hidden_size)
        self.layernorm = torch.nn.LayerNorm(embeddings.shape[-1])

        # self.linear_U = torch.nn.Linear(300, hidden_size)
        # self.linear_V = torch.nn.Linear(hidden_size, num_class)
        # self.linear_W = torch.nn.Linear(hidden_size, hidden_size)
        # self.activation = torch.nn.Tanh()
        # self.softmax = torch.nn.Softmax(dim = 1)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        pad_len = len(batch[0])
        embed_vector = self.embed(batch) # shape (batch_size, pad_len) -> (batch_size, pad_len, 300)
        normalized_embed = self.layernorm(embed_vector)
        RNN_output, hn = self.RNN(embed_vector) #  shape (batch_size, pad_len, 300) -> (batch_size, pad_len, hidden_size * D)
        fc_output = self.fc_layer(RNN_output)
        normalized_output = self.normalization(fc_output.transpose(1, 2)).transpose(1, 2)
        z = self.activation(normalized_output)
        y = self.classifier(z)

        # y = self.classifier(RNN_output) # shape (batch_size, pad_len, hidden_size * D) -> (batch_size, pad_len, num_class)
        # pad_len = len(batch[0])
        # hidden = torch.zeros(1, self.hidden_size)
        # for i in range(pad_len):
        #     Ux = self.linear_U(embed_vector[:,i,:])
        #     hidden = self.activation(self.linear_W(hidden) + Ux)
        # y = self.softmax(self.linear_V(hidden)) # shape (batch_size, num_class)
        return y
