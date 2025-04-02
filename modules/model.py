import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from memory import Memory_Unit
from translayer import Transformer
from utils import norm

from modules.MFP import mfp
from modules.temporal import Temporal
from modules.loss import QuadrupletMarginLoss
from modules.GCN import GCN
from modules.Fusion import GatedFusion


class ADCLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(), nn.Linear(128,out_dim), nn.Sigmoid())
    def forward(self, x):
        return self.mlp(x)

class DTCL(Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums

        self.fusion=GatedFusion()
        self.mfp = mfp(1024, num_layers=3, dropout_rate=0.4)
        self.tem = Temporal(1024, 512, num_layers=3)
        self.quadruplet = QuadrupletMarginLoss(margin=1.0, p=2)
        self.gcn = GCN(in_features=512, out_features=512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.5)

        self.cls_head = ADCLS_head(1024, 1)
        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)

        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        self.relu = nn.ReLU()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):

        b, t, d = x.size()
        n = 1

        s=self.tem(x)
        x = self.mfp(x,s)
        z = self.gcn(x)
        x = self.selfatt(x)
        x = self.fusion(x,z)

        if self.flag == "Train":
            N_x = x[:b*n//2]                  #### Normal part
            A_x = x[b*n//2:]                  #### Abnormal part
            A_att, A_aug = self.Amemory(A_x)   ###bt,btd,   anomaly video --->>>>> Anomaly memeory  at least 1 [1,0,0,...,1]
            N_Aatt, N_Aaug = self.Nmemory(A_x) ###bt,btd,   anomaly video --->>>>> Normal memeory   at least 0 [0,1,1,...,1]

            A_Natt, A_Naug = self.Amemory(N_x) ###bt,btd,   normal video --->>>>> Anomaly memeory   all 0 [0,0,0,0,0,...,0]
            N_att, N_aug = self.Nmemory(N_x)   ###bt,btd,   normal video --->>>>> Normal memeory    all 1 [1,1,1,1,1,...,1]
    
            _, A_index = torch.topk(A_att, t//16 + 1, dim=-1)
            negative_ax = torch.gather(A_x, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)
            
            _, N_index = torch.topk(N_att, t//16 + 1, dim=-1)
            anchor_nx=torch.gather(N_x, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            _, P_index = torch.topk(N_Aatt, t//16 + 1, dim=-1)
            positivte_nx = torch.gather(A_x, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            _, Q_index = torch.topk(A_Natt, t // 16 + 1, dim=-1)
            hard_ax = torch.gather(N_x, 1, Q_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n,-1).mean(1)

            quadruplet_margin_loss = self.quadruplet(norm(anchor_nx), norm(positivte_nx), norm(hard_ax), norm(negative_ax))

            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)
            A_aug_new = self.encoder_mu(A_aug)


            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)

            x = torch.cat((x, (torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=0))), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
    
            return {
                    "frame": pre_att,
                    'triplet_margin': quadruplet_margin_loss,
                }
        else:           
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)  

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)

            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            # print('pre',pre_att.shape)
            return {"frame": pre_att}

            # SNE
            # x = x.mean(axis=1)
            # x = x.max(dim=1)[0]

            # pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            # max_indices = pre_att.argmax(dim=1)
            # max_indices =max_indices.unsqueeze(1)
            # x = torch.gather(x, dim=1, index=max_indices.unsqueeze(2).expand(-1, -1,x.size(2))).squeeze(1) # 形状为 (b, 1024)
            #
            #
            #
            # return x

if __name__ == "__main__":
    m = DTCL(input_size = 1024, flag = "Train", a_nums = 60, n_nums = 60).cuda()
    src = torch.rand(100, 32, 1024).cuda()
    out = m(src)["frame"]
    
    print(out.size())
    
