from torch import nn 
import torch
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AdaHyperedgeGen(nn.Module):

    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)  
        elif context == "both":
            self.context_net = nn.Linear(2*node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(
                f"Unsupported context '{context}'. "
                "Expected one of: 'mean', 'max', 'both'."
            )

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
    
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)          
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)          
        else:
            avg_context = X.mean(dim=1)           
            max_context, _ = X.max(dim=1)           
            context_cat = torch.cat([avg_context, max_context], dim=-1) 
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)  
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets           
        
        X_proj = self.pre_head_proj(X) 
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling 
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1) 
        
        logits = self.dropout(logits)  

        return F.softmax(logits, dim=1)
    

class AdaHGConv(nn.Module):

    def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, dropout=0.05, context="both"):
        super().__init__()
        self.edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        
    def forward(self, X):
        A = self.edge_generator(X)  
        
        He = torch.bmm(A.transpose(1, 2), X) 
        He = self.edge_proj(He)
        
        X_new = torch.bmm(A, He)  
        X_new = self.node_proj(X_new)
        
        return X_new + X

def column_batching(tensor, batch_size):

    H, W = tensor.shape
    
    # 计算完整的批次数量
    num_batches = W // batch_size
    
    # 截取可以整除的部分
    batches = []
    for i in range(num_batches):
        start_col = i * batch_size
        end_col = (i + 1) * batch_size
        batch = tensor[:, start_col:end_col]
        batches.append(batch)
    
    # 如果有剩余的列，创建一个单独的批次
    if W % batch_size != 0:
        remaining = tensor[:, num_batches * batch_size:]
        # 如果剩余列数不足，用零填充
        if remaining.shape[1] < batch_size:
            padding = torch.zeros(H, batch_size - remaining.shape[1], 
                                  dtype=tensor.dtype, device=tensor.device)
            remaining = torch.cat([remaining, padding], dim=1)
        batches.append(remaining)
    
    # 在第0维度堆叠所有批次
    batched_tensor = torch.stack(batches, dim=0)
    
    return batched_tensor


def column_unbatching(batched_tensor, original_width=None):

    B, H, C = batched_tensor.shape
    
    # 如果没有提供原始宽度，则计算可能的最大宽度
    if original_width is None:
        original_width = B * C
    
    # 将批次维度展平
    reconstructed = batched_tensor.permute(1, 0, 2).reshape(H, -1)
    
    # 截取原始宽度
    if reconstructed.shape[1] > original_width:
        reconstructed = reconstructed[:, :original_width]
    
    return reconstructed


class HyperConv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, std=0.01):
        super(HyperConv, self).__init__()
        self.weight_trans = Parameter(torch.Tensor(in_ft, out_ft))
        nn.init.xavier_normal_(self.weight_trans.data, gain=std)
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
            nn.init.normal_(self.bias, mean=0, std=std)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, X, H):
        X = torch.mm(X, self.weight_trans)
        DV = torch.sum(H, dim=1).float()
        DV += 1e-12
        DE = torch.sum(H, dim=0).float()
        DE += 1e-12
        invDE = torch.diag(DE.pow(-1))
        HT = H.T
        DV1 = torch.diag(DV.pow(-1))
        G = DV1 @ H @ invDE @ HT
        # DV2 = torch.diag(DV.pow(-0.5))
        # G = DV2 @ H @ invDE @ HT @ DV2
        X = G.matmul(X)
        if self.bias is not None:
            X = X + self.bias
        else:
            pass
        X = F.relu(X)
        return X
    

class Qnet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Qnet, self).__init__()
        self.alpha0 = nn.Linear(1, 1, bias=True)
        self.alpha1 = nn.Linear(in_dim, in_dim)
        self.conv0 = HyperConv(in_dim, in_dim)

        self.alpha2 = nn.Linear(1, 1, bias=True)
        self.alpha3 = nn.Linear(in_dim, in_dim)
        self.conv1 = HyperConv(in_dim, in_dim)        

        self.adaConv0 = AdaHGConv(8)


        self.alpha4 = nn.Linear(1, 1, bias=True)
        self.alpha5 = nn.Linear(in_dim, in_dim)
        self.conv2 = HyperConv(in_dim, in_dim) 

        self.alpha6 = nn.Linear(1, 1, bias=True)
        self.alpha7 = nn.Linear(in_dim, in_dim)
        self.conv3 = HyperConv(in_dim, in_dim) 

        self.adaConv1 = AdaHGConv(8)

        self.alpha6 = nn.Linear(1, 1, bias=True)
        self.alpha7 = nn.Linear(in_dim, in_dim)
        self.conv4 = HyperConv(in_dim, in_dim)

        self.conv5 = HyperConv(in_dim, in_dim) 
        self.adaConv2 = AdaHGConv(8)


        self.alpha8 = nn.Linear(1, 1, bias=True)
        self.alpha9 = nn.Linear(in_dim, in_dim)
        self.conv6 = HyperConv(in_dim, in_dim)

        self.conv7 = HyperConv(in_dim, in_dim) 
        self.adaConv3 = AdaHGConv(8)

        self.fc = nn.Linear(in_dim * 8, out_dim)

    def forward(self, X, H, states):
        
        segments = torch.split(X, 32, dim=1)
        
        X_o_h = segments[0]        
        X_d_h = segments[1]     
        X_d_l = segments[2]     
        X_o_l = segments[3]        

        X_o_l_0 = F.leaky_relu(self.alpha1(X_o_l + self.alpha0(states.view(-1, 1))), 0.2)
        X_o_l_1 = self.conv0(X_o_l_0, H)

        X_o_l_2 = F.leaky_relu(self.alpha3(X_o_l_1 + self.alpha2(states.view(-1, 1))), 0.2)
        X_o_l_3 = self.conv1(X_o_l_2, H)

        X_o_l_4 = column_batching(X_o_l_3, 8)
        X_o_l_5 = self.adaConv0(X_o_l_4)
        X_o_l_6 = column_unbatching(X_o_l_5)

        X_o_h_0 = F.leaky_relu(self.alpha5(X_o_h + self.alpha4(states.view(-1, 1))), 0.2)
        X_o_h_1 = self.conv2(X_o_h_0, H)

        X_o_h_2 = F.leaky_relu(self.alpha7(X_o_h_1 + self.alpha6(states.view(-1, 1))), 0.2)
        X_o_h_3 = self.conv3(X_o_h_2, H)

        X_o_h_4 = column_batching(X_o_h_3, 8)
        X_o_h_5 = self.adaConv1(X_o_h_4)
        X_o_h_6 = column_unbatching(X_o_h_5)


        X_d_h = H.T @ F.leaky_relu(self.alpha7(X_d_h + self.alpha6(states.view(-1, 1))), 0.2)
        X_d_h_0 = self.conv4(X_d_h, H.T)
        X_d_h_1 = self.conv5(X_d_h_0, H.T)

        X_d_h_2 = column_batching(X_d_h_1, 8)
        X_d_h_3 = self.adaConv2(X_d_h_2)
        X_d_h_4 = column_unbatching(X_d_h_3)

        X_d_l = H.T @ F.leaky_relu(self.alpha9(X_d_l + self.alpha8(states.view(-1, 1))), 0.2)
        X_d_l_0 = self.conv6(X_d_l, H.T)
        X_d_l_1 = self.conv7(X_d_l_0, H.T)

        X_d_l_2 = column_batching(X_d_l_1, 8)
        X_d_l_3 = self.adaConv3(X_d_l_2)
        X_d_l_4 = column_unbatching(X_d_l_3)

        X = torch.cat([X_o_l_6, X_o_l_3, X_o_h_6, X_o_h_3, H @ X_d_h_1 ,H @ X_d_h_4, H @ X_d_l_1, H @ X_d_l_4], dim=1)
        # print(X.shape)

        return self.fc(X)


class DualHGCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DualHGCN, self).__init__()
        self.conv_0 = HyperConv(in_dim, out_dim)
        self.conv_1 = HyperConv(out_dim, out_dim)
        # self.conv_2 = HyperConv(out_dim, out_dim)
        self.adaConv_0 = AdaHGConv(8)
        self.adaConv_1 = AdaHGConv(8)

        self.conv_0_d = HyperConv(in_dim, out_dim)
        self.conv_1_d = HyperConv(out_dim, out_dim)
        # self.conv_2_d = HyperConv(out_dim, out_dim)
        self.adaConv_d_0 = AdaHGConv(8)
        self.adaConv_d_1 = AdaHGConv(8)

        self.fc = nn.Linear(128, 1)


    def forward(self, X, X_d, H):
        # DV = torch.sum(H, dim=1)
        # DV += 1e-12
        # DV1 = torch.diag(DV.pow(-1))

        # DE = torch.sum(H, dim=0)
        # DE += 1e-12
        
        X_0 = self.conv_0(X, H)
        X_0_d = self.conv_0_d(X_d, H.T)

        # X_0 = X_0 + H @ X_0_d
        # X_0_d = X_0_d + H.T @ X_0

        X_1 = self.conv_1(X_0, H)
        X_1_d = self.conv_1_d(X_0_d, H.T)

        # X_1 = X_1 + H @ X_1_d
        # X_1_d = X_1_d + H.T @ X_1

        # X_2 = self.conv_2(X_1, H)
        X_2 = column_batching(X_1, 8)
        X_3 = self.adaConv_0(X_2)
        X_3 = column_unbatching(X_3)
        
        
        # X_2_d = self.conv_2_d(X_1_d, H.T)
        X_2_d = column_batching(X_1_d, 8)
        X_3_d = self.adaConv_d_0(X_2_d)
        X_3_d = column_unbatching(X_3_d)

        # torch.Size([565])
        # torch.Size([565, 601])
        # torch.Size([601, 32])
        # torch.Size([32])
        # torch.Size([32])

        # print(DV.shape)
        # print(H.shape)
        # print(X_3_d.shape)
        # print((DV @ H @ X_3_d).shape)
        # print((DV @ H @ X_1_d).shape)
        
        
        X_final = torch.cat([X_3,  H @ X_3_d,  H @ X_1_d, X_1], dim=1)
        
        return self.fc(X_final), X_final
        