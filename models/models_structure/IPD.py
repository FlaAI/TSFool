from torch import nn


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(128, 2)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state
        out = self.out(r_out[:, -1, :])        # choose r_out at the last time step
        out_trace = self.out(r_out)            # choose r_out at all time steps
        return out, out_trace
