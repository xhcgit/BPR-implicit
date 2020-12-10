import torch as t
import torch.nn as nn
import numpy as np

class Test(nn.Module):
    def __init__(self, userNum, itemNum, hide_dim):
        super(Test, self).__init__()
        self.userEmbed = nn.Embedding(userNum, hide_dim)
        self.itemEmbed = nn.Embedding(itemNum, hide_dim)
        nn.init.xavier_normal_(self.userEmbed.weight)
        nn.init.xavier_normal_(self.itemEmbed.weight)

    def forward(self, user_idx, item_idx):
        i_e = self.itemEmbed(item_idx)
        u_e = self.userEmbed(user_idx)

        prediction = t.sum(u_e*i_e, 1)
        return prediction


