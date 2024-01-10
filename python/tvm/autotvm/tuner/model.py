import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class Encoder(torch.nn.Module):
    def __init__(
        self,
        num_layers=3,
        embed_size=752,
        num_heads=16,
        dropout=0.2,
        hidden_size=256,
        output_size=1,
    ):
        super().__init__()
        """
        原本实现
        """
        self.layer = torch.nn.TransformerEncoderLayer(embed_size, num_heads, hidden_size, dropout)
        self.encoder = torch.nn.TransformerEncoder(self.layer, num_layers)
        self.decoder = torch.nn.Linear(embed_size, output_size)
        self.src_mask = None

        """
        两层TransformerEncoder+一层双向LSTM
        """
        # self.linear = nn.Sequential(
        #     torch.nn.Linear(embed_size, hidden_size),
        #     torch.nn.ReLU(),
        # )
        # self.layer = torch.nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        # self.encoder = torch.nn.TransformerEncoder(self.layer, 2)
        # self.lstm = torch.nn.LSTM(embed_size, embed_size, num_layers=1, bidirectional=True)
        # self.decoder = torch.nn.Linear(hidden_size, output_size)
        # self.src_mask = None

        # self.decoderlayer = torch.nn.TransformerDecoderLayer(embed_size, num_heads, hidden_size, dropout)
        # self.tfdecoder = torch.nn.TransformerDecoder(self.decoderlayer, num_layers)


    def forward(self, inputs):
        """
        原本实现
        """
        encode = self.encoder(inputs, self.src_mask)
        output = self.decoder(encode)
        return output.squeeze()

        """
        两层TransformerEncoder+一层双向LSTM
        """
        # output = self.linear(inputs)
        # output = self.encoder(output, self.src_mask)
        # # output, (h, c) = self.lstm(encode)
        # output = self.decoder(output)
        # return output.squeeze()

        # output = self.tfdecoder(inputs, inputs)
        # output = self.decoder(output)
        # return output.squeeze()

class LSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        """
        原本实现
        """
        # self.lstm1 = torch.nn.LSTM(in_dim, hidden_dim, num_layers=3)
        # self.decoder = torch.nn.Linear(hidden_dim, out_dim)

        self.norm = torch.nn.Identity()

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mha = torch.nn.MultiheadAttention(hidden_dim, 4)

        self.lstm = nn.LSTM(input_size=hidden_dim,
                               hidden_size=hidden_dim,
                               num_layers=3,
                               batch_first=False,
                               dropout=0,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(
            2*hidden_dim, 2*hidden_dim))
        self.u_omega = nn.Parameter(torch.Tensor(2*hidden_dim, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.lstm_decoder = nn.Linear(2*hidden_dim, out_dim)
        self.mlp_decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, output):
        """
        用于原本和双向LSTM的前向传播
        """
        # output, (h, c) = self.lstm1(output)
        # output = self.decoder(output).squeeze()
        # return output

        """
        用于TenSet的LSTM实现的前向传播
        """
        # output = self.encoder(output, self.src_mask)
        # output, (h, c) = self.lstm(output)
        # output = self.norm(h[0])
        # output = self.linear(output) + output
        # output = self.linear(output) + output
        # output = self.decoder(output)
        # return output
        """
        双向LSTM+attention
        """
        output = self.linear(output)
        # output = self.encoder(output, self.src_mask)

        mlp_output = self.mlp(output)
        # mlp_output = self.mha(self.q(mlp_output), self.k(mlp_output), self.v(mlp_output))[0] + mlp_output

        lstm_output, (h, c) = self.lstm(output)
        # output形状是(seq_len,batch_size, 2 * num_hiddens)
        x = lstm_output.permute(1, 0, 2)
        # x形状是(batch_size, seq_len, 2 * num_hiddens)

        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束

        lstm_output = torch.sum(scored_x, dim=1)
        # # # feat形状是(batch_size, 2 * num_hiddens)
        # # # outs = self.decoder(feat)
        # # # out形状是(batch_size, 1)

        # output = self.norm(feat)
        # # # output = self.linear(output) + output
        # output = self.linear2(output) + output
        # output = self.linear2(output) + output
        lstm_output = self.lstm_decoder(lstm_output)
        mlp_output = self.mlp_decoder(mlp_output)
        output = lstm_output + mlp_output
        return output.squeeze()

class TPP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        """
        原本实现
        """
        # self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        # self.relu1 = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.relu2 = torch.nn.ReLU()
        # self.linear3 = torch.nn.Linear(hidden_dim, out_dim)

        """
        TLP结构
        """
        # self.relu = torch.nn.ReLU()
        # self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        # self.linearHidden = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.MHA = torch.nn.MultiheadAttention(hidden_dim, 4)
        # self.decoder = torch.nn.Linear(hidden_dim, out_dim)
        # self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.lstm = nn.LSTM(input_size=hidden_dim,
                               hidden_size=hidden_dim,
                               num_layers=3,
                               batch_first=False,
                               dropout=0,
                               bidirectional=True)
        self.w_omega = nn.Parameter(torch.Tensor(2*hidden_dim, 2*hidden_dim))
        self.u_omega = nn.Parameter(torch.Tensor(2*hidden_dim, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
    
        self.forwardNN = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.lstm_decoder = nn.Linear(2*hidden_dim, hidden_dim)
        self.mlp_decoder = nn.Linear(hidden_dim, hidden_dim)

        self.linearHidden = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.reg = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, output):
        # output = self.linear1(output)
        # output = self.relu1(output)
        # output = self.linear2(output)
        # output = self.relu2(output)
        # output = self.linear3(output)
        # return output.squeeze()

        """
        TLP结构
        """
        # output = self.linear1(output)
        # output = self.relu(output)
        # output = self.linearHidden(output)
        # output = self.relu(output)
        # output = self.linearHidden(output)
        # output = self.relu(output)
        # output = self.linearHidden(output)
        # output = self.relu(output)

        # identity = output
        # output = self.MHA(self.q(output), self.k(output), self.v(output))[0]
        # output += identity
        # # output = self.relu(output)

        # identity = output
        # output = self.linearHidden(output)
        # output = self.relu(output)
        # output += identity
        # output = self.relu(output)

        # identity = output
        # output = self.linearHidden(output)
        # output = self.relu(output)
        # output += identity
        # output = self.relu(output)

        # output = self.linearHidden(output)
        # output = self.relu(output)
        # output = self.linearHidden(output)
        # output = self.relu(output)
        # output = self.linearHidden(output)
        # output = self.relu(output)

        # output = self.decoder(output)
        # return output.squeeze()

        output = self.forwardNN(output)

        lstm_output, (h, c) = self.lstm(output)
        x = lstm_output.permute(1, 0, 2)
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        lstm_output = torch.sum(scored_x, dim=1)

        mlp_output = self.mlp(output)

        lstm_output = self.lstm_decoder(lstm_output)
        mlp_output = self.mlp_decoder(mlp_output)
        output = lstm_output + mlp_output
        output = self.linearHidden(output) + output
        output = self.linearHidden(output) + output
        output = self.linearHidden(output) + output
        output = self.reg(output)
        return output.squeeze()

class TPP_LSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
    
        self.lstm = nn.LSTM(input_size=hidden_dim,
                               hidden_size=hidden_dim,
                               num_layers=3,
                               batch_first=False,
                               dropout=0,
                               bidirectional=True)
        self.w_omega = nn.Parameter(torch.Tensor(2*hidden_dim, 2*hidden_dim))
        self.u_omega = nn.Parameter(torch.Tensor(2*hidden_dim, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
    
        self.forwardNN = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.lstm_decoder = nn.Linear(2*hidden_dim, hidden_dim)
        self.mlp_decoder = nn.Linear(hidden_dim, hidden_dim)

        self.linearHidden = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.reg = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, output):
        output = self.forwardNN(output)

        lstm_output, (h, c) = self.lstm(output)
        x = lstm_output.permute(1, 0, 2)
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        lstm_output = torch.sum(scored_x, dim=1)

        output = self.lstm_decoder(lstm_output)
        output = self.linearHidden(output) + output
        output = self.linearHidden(output) + output
        output = self.linearHidden(output) + output
        output = self.reg(output)
        return output.squeeze()

class TPP_MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.lstm = nn.LSTM(input_size=hidden_dim,
                               hidden_size=hidden_dim,
                               num_layers=3,
                               batch_first=False,
                               dropout=0,
                               bidirectional=True)
        self.w_omega = nn.Parameter(torch.Tensor(2*hidden_dim, 2*hidden_dim))
        self.u_omega = nn.Parameter(torch.Tensor(2*hidden_dim, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
    
        self.forwardNN = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.lstm_decoder = nn.Linear(2*hidden_dim, hidden_dim)
        self.mlp_decoder = nn.Linear(hidden_dim, hidden_dim)

        self.linearHidden = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.reg = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, output):
        output = self.forwardNN(output)
        mlp_output = self.mlp(output)
        output = self.mlp_decoder(mlp_output)
        output = self.linearHidden(output) + output
        output = self.linearHidden(output) + output
        output = self.linearHidden(output) + output
        output = self.reg(output)
        return output.squeeze()