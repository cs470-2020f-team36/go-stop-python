"""
network.py

Network definition.
"""

import torch
from torch import Tensor
import torch.nn as nn

from go_stop.train.args import args

# pylint: disable=invalid-name


class MLPBlock(nn.Module):
    """
    Implementation of an MLP block
    consisting of a fully connected layer,
    a batch normalization, and an activation function ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: Tensor):
        """Forward the input into the output."""

        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class OutBlock(nn.Module):
    """
    Implementation of the output block.

    For the policy: it is a composition of two MLP layers and a softmax function.
    For the value: it is a composition of an MLP layer, and a summation of MLP and tanh(MLP).
    """
    def __init__(self, in_channels: int):
        super().__init__()

        self.p_fc1 = MLPBlock(in_channels, args.nhid)
        self.p_fc2 = MLPBlock(args.nhid, args.nout)

        self.v_fc1 = MLPBlock(in_channels, args.nhid)
        self.v_fc2 = MLPBlock(args.nhid, 1)
        self.v_fc3 = MLPBlock(args.nhid, 1)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: Tensor):
        """Forward the input into the output."""

        policy = self.p_fc1(x)
        policy = self.p_fc2(policy)
        policy = self.log_softmax(policy)
        policy = policy.exp()

        value = self.v_fc1(x)
        value = self.v_fc2(value) + torch.tanh(self.v_fc3(value))
        value = value.squeeze()

        return policy, value


class EncoderNet(nn.Module):
    """The whole network"""

    def __init__(self):
        super().__init__()

        self.fc = MLPBlock(args.ninp, args.nhid)
        self.layers = torch.nn.Sequential(
            *[MLPBlock(args.nhid, args.nhid) for _ in range(6)]
        )
        self.out = OutBlock(args.nhid)

    def forward(self, x):
        """Forward the input into the output."""

        x = self.fc(x)
        x = self.layers(x)

        return self.out(x)


class AlphaLoss(nn.Module):
    """Loss criterion of the network."""

    # pylint: disable=no-self-use
    def forward(self, pred, target):
        """
        `pred` and `target` are both of type (policy, value), where
            + policy: a tensor of shape (B, args.nout),
            + value: a tensor of shape (B,).
        """

        policy_pred, value_pred = pred
        policy_target, value_target = target

        # policy_loss(s) = − \sum_{a: action} -p_{target}(s, a) \log(pi_{pred}(s, a))
        policy_loss = torch.sum(
            (-policy_target * (1e-8 + policy_pred.float()).float().log()),
            dim=1,
        )

        # value_loss(s) = (v_{pred}(s) − v_{target}(s))^2
        value_loss = ((value_target - value_pred) ** 2).view(-1).float()

        # loss(s) = policy_loss(s) + value_loss(s)
        total_loss = (policy_loss + value_loss).mean()

        return total_loss
