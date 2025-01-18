import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .convlstm import (
    MultiLayerCustomLSTM,  # <- Import our new multi-layer linear LSTM
)


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ModelFreeACModel(nn.Module):
    def __init__(self, obs_space, action_space, recurrent=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.recurrent = recurrent

        # You can adjust how many layers we stack in MultiLayerLinearLSTM
        self.num_recurrent_layers = 3  # Example: 2 layers

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),  # -1
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # // 2
            nn.Conv2d(16, 32, (2, 2)),  # -1
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),  # -1
            nn.ReLU(),
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        # If recurrent, use our multi-layer linear LSTM (with run_n_ticks).
        if self.recurrent:
            # The hidden dim is the same as our “semi_memory_size”
            self.hidden_dim = self.semi_memory_size
            # Create our multi-layer linear LSTM
            self.memory_rnn = MultiLayerCustomLSTM(
                input_dim=self.image_embedding_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_recurrent_layers,
            )
            # You can choose how many repeated ticks per input step you want
            self.n_ticks = 3  # Example: run each input through multiple ticks.

        # (Optional) define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(
                obs_space["text"], self.word_embedding_size
            )
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(
                self.word_embedding_size, self.text_embedding_size, batch_first=True
            )

        # Combine image embedding with text embedding if needed
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        # Initialize parameters
        self.apply(init_params)

    @property
    def semi_memory_size(self):
        # By default, we match old code's approach: image_embedding_size as memory dimension.
        return self.image_embedding_size

    @property
    def memory_size(self):
        # If we have num_recurrent_layers D, each layer has a c and h of size hidden_dim
        # => total = 2 * hidden_dim * num_recurrent_layers
        return 2 * self.hidden_dim * self.num_recurrent_layers

    def forward(self, obs, memory=None):
        """
        obs    : Dict of observations, should include obs.image
        memory : shape (batch, memory_size) or None
                 memory is [c_1, h_1, c_2, h_2, ..., c_D, h_D] concatenated along dim=1
        """
        # Build the image embedding
        x = obs.image.transpose(1, 3).transpose(2, 3)  # (batch, 3, H, W)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)  # flatten

        # Optionally handle text embedding
        if self.use_text:
            x_text = self._get_embed_text(obs.text)
            x = torch.cat((x, x_text), dim=1)

        if self.recurrent:
            # Unpack memory into c/h states
            c_prev_list, h_prev_list = self.unpack_memory(memory)

            # Now call run_n_ticks for each input in the batch
            # We pass the same x for each tick, but you can adapt if needed
            c_curr_list, h_curr_list = self.memory_rnn.run_n_ticks(
                x, c_prev_list, h_prev_list, n_ticks=self.n_ticks
            )

            # The top layer’s hidden state
            # If we have D layers, the last hidden is h_curr_list[-1]
            embedding = h_curr_list[-1]

            # Re-pack states for next step
            memory = self.pack_memory(c_curr_list, h_curr_list)
        else:
            # If not recurrent, simply use x as the embedding
            embedding = x

        # Actor: categorical distribution
        logits = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(logits, dim=1))

        # Critic: value output
        value = self.critic(embedding)
        value = value.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        # text is (batch, max_text_length)
        _, hidden = self.text_rnn(self.word_embedding(text))
        # hidden is (num_layers, batch, hidden_dim); take last layer
        return hidden[-1]

    def pack_memory(self, c_list, h_list):
        # c_list, h_list are lists each of length num_recurrent_layers
        # Each c_i,h_i has shape (batch, hidden_dim)
        # We want to return shape (batch, 2 * hidden_dim * num_layers)
        # ordering: c_1, h_1, c_2, h_2, ...
        mems = []
        for c_i, h_i in zip(c_list, h_list):
            mems.append(c_i)
            mems.append(h_i)
        return torch.cat(mems, dim=1)

    def unpack_memory(self, memory):
        """
        memory : shape (batch, 2 * hidden_dim * num_layers)
        returns c_list, h_list lists of length num_layers
        """
        # Each layer has c_i,h_i => 2 * hidden_dim
        bs = memory.shape[0]
        c_list = []
        h_list = []
        offset = 0
        for _layer in range(self.num_recurrent_layers):
            c_i = memory[:, offset : offset + self.hidden_dim]
            h_i = memory[:, offset + self.hidden_dim : offset + 2 * self.hidden_dim]
            offset += 2 * self.hidden_dim

            c_list.append(c_i)
            h_list.append(h_i)

        return c_list, h_list
