import torch
import torch.nn as nn


class CustomLSTMCell(nn.Module):
    """
    A drop-in replacement for the ConvLSTM cell equations, but using
    Linear layers in place of convolutions.

    Equations for layer d > 1:
      f_d^n = sigma(W_fi * i_t + W_fh1 * h_d^{n-1} + W_fh2 * h_{d-1}^n + b_f)
      i_d^n = sigma(W_ii * i_t + W_ih1 * h_d^{n-1} + W_ih2 * h_{d-1}^n + b_i)
      o_d^n = sigma(W_oi * i_t + W_oh1 * h_d^{n-1} + W_oh2 * h_{d-1}^n + b_o)

      c_d^n = f_d^n ⊙ c_d^{n-1} +
              i_d^n ⊙ tanh(W_ci * i_t + W_ch1 * h_d^{n-1} + W_ch2 * h_{d-1}^n + b_c)

      h_d^n = o_d^n ⊙ tanh(c_d^n)

    For d = 1, we usually replace h_{d-1}^n with h_{D}^{n-1} (the “top‐down” skip).
    We handle that by passing in skip_h_above as the correct “h_{d-1}^n” input.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # -- Forget gate: f_d^n
        self.W_f_i = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_f_h1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_f_h2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # -- Input gate: i_d^n
        self.W_i_i = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_i_h1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_i_h2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # -- Output gate: o_d^n
        self.W_o_i = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_o_h1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o_h2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # -- Cell update candidate: tanh(...)
        self.W_c_i = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_c_h1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_c_h2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, i_t, c_prev, h_prev, skip_h_above):
        """
        i_t          : current input at time t       (batch, input_dim)
        c_prev       : previous cell state c_d^{n-1} (batch, hidden_dim)
        h_prev       : previous hidden h_d^{n-1}     (batch, hidden_dim)
        skip_h_above : top-down or skip connection
                       (for d>1 this is h_{d-1}^n;
                        for d=1 this may be h_{D}^{n-1})  (batch, hidden_dim)
        Returns:
            c_new, h_new
        """

        # Forget gate
        f = torch.sigmoid(
            self.W_f_i(i_t) + self.W_f_h1(h_prev) + self.W_f_h2(skip_h_above)
        )

        # Input gate
        i = torch.sigmoid(
            self.W_i_i(i_t) + self.W_i_h1(h_prev) + self.W_i_h2(skip_h_above)
        )

        # Output gate
        o = torch.sigmoid(
            self.W_o_i(i_t) + self.W_o_h1(h_prev) + self.W_o_h2(skip_h_above)
        )

        # Candidate cell update
        g = torch.tanh(
            self.W_c_i(i_t) + self.W_c_h1(h_prev) + self.W_c_h2(skip_h_above)
        )

        # New cell state
        c_new = f * c_prev + i * g

        # New hidden
        h_new = o * torch.tanh(c_new)

        return c_new, h_new


class MultiLayerCustomLSTM(nn.Module):
    """
    Builds D layers of the above LinearLSTMCell.  For each time step n:
      - if d=1, use skip = h_{D}^{n-1}
      - if d>1, use skip = h_{d-1}^n
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # For the first layer, input dim is 'input_dim'
        self.cells = nn.ModuleList()
        self.cells.append(CustomLSTMCell(input_dim, hidden_dim))

        # For subsequent layers, the input dims might differ.
        # But here we keep it simple: each layer uses the same hidden_dim for h_{d-1}^n
        for _ in range(1, num_layers):
            # If you want each layer to accept the "input_dim" as well,
            #   you can adapt. We'll keep it the same for simplicity.
            self.cells.append(CustomLSTMCell(input_dim, hidden_dim))

    def forward(self, i_t, c_prev_list, h_prev_list):
        """
        i_t          : input at time t, shape (batch, input_dim)
        c_prev_list  : list of length D of previous cell states [c_1^{n-1}, ..., c_D^{n-1}]
        h_prev_list  : list of length D of previous hidden states [h_1^{n-1}, ..., h_D^{n-1}]

        Returns:
           c_new_list, h_new_list  (the updated cell and hidden states)
        """
        # We will store the updated states in these lists
        c_new_list = [None] * self.num_layers
        h_new_list = [None] * self.num_layers

        # We'll also need the "top-down skip" from the D-th layer's old hidden
        # for layer d=1:
        top_down_skip = h_prev_list[-1]  # h_{D}^{n-1}

        for d in range(self.num_layers):
            if d == 0:
                # For the first layer, skip = top_down_skip
                skip_h_above = top_down_skip
            else:
                # For higher layers, skip = h_{d-1}^n
                skip_h_above = h_new_list[d - 1]

            c_new, h_new = self.cells[d](
                i_t, c_prev_list[d], h_prev_list[d], skip_h_above
            )
            c_new_list[d] = c_new
            h_new_list[d] = h_new

        return c_new_list, h_new_list

    def run_n_ticks(self, i_t, c_prev_list, h_prev_list, n_ticks):
        """
        Repeatedly update the LSTM stack for n_ticks steps,
        using the same input i_t each time.

        Args:
        i_t          : (batch, input_dim) — the single input for all ticks
        c_prev_list  : list of [c_1^{prev}, ..., c_D^{prev}] before the first tick
        h_prev_list  : list of [h_1^{prev}, ..., h_D^{prev}] before the first tick
        n_ticks      : integer, number of times to update

        Returns:
        c_curr_list, h_curr_list : the final states after n_ticks
        """
        c_curr_list = c_prev_list
        h_curr_list = h_prev_list

        for _ in range(n_ticks):
            c_curr_list, h_curr_list = self.forward(i_t, c_curr_list, h_curr_list)

        return c_curr_list, h_curr_list
