import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

#Resblock
class ResBlock(nn.Module):
    '''
    Input: [B, inC, H, W]
    Output: [B, outC, H*, W*]
    '''
    def __init__(self, in_channels, out_channels, mode = "normal", time_emb_size=512):
        #mode could be 'normal', 'down', 'up'
        super(ResBlock, self).__init__()
        self.mode = mode
        self.groupnorm1 = nn.GroupNorm(num_groups = 32, num_channels = in_channels)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size= 3, padding=1)
        
        self.linear = nn.Linear(in_features = time_emb_size, out_features = out_channels)
        
        self.groupnorm2 = nn.GroupNorm(num_groups = 32, num_channels = out_channels)

        #for downsampling
        self.avg = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        
        self.dropout = nn.Dropout2d(p = 0.5)

        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size= 3, padding=1)
        
           
    def forward(self, feature, time):
        #feature is [B, C, H, W]
        #time is [1, time_emb_size]

        out_feature = self.groupnorm1(feature) #[B, inC, H, W]
        out_feature = F.silu(out_feature) #[B, inC, H, W]

        if self.mode == "normal":
            out_feature = self.conv1(out_feature) #[B, outC, H, W]
        elif self.mode == "down":
            out_feature = self.avg(out_feature) #[B, outC, H/2, W/2]
        elif self.mode == "up":
            out_feature = F.interpolate(out_feature, scale_factor=2, mode='bilinear') #[B, outC, 2H, 2W]

        out_time = F.silu(time) #[1, time_emb_size]
        out_time = self.linear(out_time) #[1, outC]
        out_time = out_time.unsqueeze(dim = -1).unsqueeze(dim = -1) #[1, outC, 1, 1]

        out = out_feature + out_time #[B, outC, H*, W*] 

        residual = out
        out = self.groupnorm2(out) #[B, outC, H*, W*] 
        out = F.silu(out) #[B, outC, H*, W*] 
        out = self.dropout(out) #[B, outC, H*, W*] 
        out = self.conv2(out) #[B, outC, H*, W*] 
        
        return out + residual


#FeedForwardNetwork
class FeedForwardNetwork(nn.Module):
    '''
    Input: [B, C, H, W]
    Output: [B, C, H, W]
    '''
    def __init__(self, dim, mult=4):
        super().__init__()

        inner_dim = int(dim * mult)

        self.net = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(), nn.Linear(inner_dim, dim))

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    '''
    Input: [B, C, H, W]
    Output: [B, C, H, W]
    '''
    def __init__(self, query_dim, heads, dim_head, context_dim=None):
        super().__init__()

        inner_dim = dim_head * heads
        
        if context_dim is None:
            context_dim = query_dim
        # print('context dim:', context_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.Q = nn.Linear(query_dim, inner_dim, bias=False)
        self.K = nn.Linear(context_dim, inner_dim, bias=False)
        self.V = nn.Linear(context_dim, inner_dim, bias=False)

        self.O = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None):
        #If context is none, it acts as self attention
        h = self.heads

        q = self.Q(x)
        
        if context is None:
            context = x
        
        k = self.K(context)
        v = self.V(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h),(q, k, v))

        aij = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = aij.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        out = self.O(out)

        return out


#Transformer block
class TransformerBlock(nn.Module):
    '''
    Input: [B, C, H, W]
    Output: [B, C, H, W]
    '''
    def __init__(self, dim, num_heads, dim_head, context_dim=None):
        super().__init__()

        #Self Attention
        self.attn1 = Attention(query_dim = dim, heads = num_heads, dim_head = dim_head)  
        
        #Cross Attention
        self.attn2 = Attention(query_dim=dim, context_dim=context_dim, heads=num_heads, dim_head=dim_head)  # is self-attn if context is none
        
        #FFN
        self.ff = FeedForwardNetwork(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.norm1(self.attn1(x)) + x
        x = self.norm2(self.attn2(x, context=context)) + x
        x = self.norm3(self.ff(x)) + x
        return x

#Spatial transformer
class SpatialTransformer(nn.Module):
    '''
    Input: [B, C, H, W] and [B, 11, 64] (slots)
    Output: [B, C, H, W]
    '''

    def __init__(self, in_channels, num_heads, dim_head = 32, depth=1, context_dim=192):
        super().__init__()

        self.in_channels = in_channels
        inner_dim = num_heads * dim_head
        self.norm = nn.GroupNorm(num_groups = 32, num_channels = in_channels)

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = inner_dim, kernel_size= 1, stride= 1, padding=0)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(inner_dim, num_heads, dim_head, context_dim= context_dim) for d in range(depth)
        ])

        self.conv2 = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        
        b, c, h, w = x.shape
        x = self.norm(x)

        x = self.conv1(x)

        x = rearrange(x, 'b c h w -> b (h w) c')
        
        for block in self.transformer_blocks:
            x = block(x, context=context)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        x = self.conv2(x)

        return x 


class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1)
        self.R1 = ResBlock(in_channels = 64, out_channels = 64)
        self.R2 = ResBlock(in_channels = 64, out_channels = 64)
        self.D1 = ResBlock(in_channels = 64, out_channels = 64, mode = "down")
        self.R3 = ResBlock(in_channels = 64, out_channels = 128)
        self.T1 = SpatialTransformer(in_channels = 128, num_heads = 4)
        self.R4 = ResBlock(in_channels = 128, out_channels = 128)
        self.T2 = SpatialTransformer(in_channels = 128, num_heads = 4)
        self.D2 = ResBlock(in_channels = 128, out_channels = 128, mode = "down")
        self.R5 = ResBlock(in_channels = 128, out_channels = 192)
        self.T3 = SpatialTransformer(in_channels = 192, num_heads = 6)
        self.R6 = ResBlock(in_channels = 192, out_channels = 192)
        self.T4 = SpatialTransformer(in_channels = 192, num_heads = 6)
        self.D3 = ResBlock(in_channels = 192, out_channels = 192, mode = "down")
        self.R7 = ResBlock(in_channels = 192, out_channels = 256)
        self.T5 = SpatialTransformer(in_channels = 256, num_heads = 8)
        self.R8 = ResBlock(in_channels = 256, out_channels = 256)
        self.T6 = SpatialTransformer(in_channels = 256, num_heads = 8)

        self.R9 = ResBlock(in_channels = 256, out_channels = 256)
        self.T7 = SpatialTransformer(in_channels = 256, num_heads = 8)
        self.R10 = ResBlock(in_channels = 256, out_channels = 256)

        self.R11 = ResBlock(in_channels = 512, out_channels = 256)
        self.T8 = SpatialTransformer(in_channels = 256, num_heads = 8)
        self.R12 = ResBlock(in_channels = 512, out_channels = 256)
        self.T9 = SpatialTransformer(in_channels = 256, num_heads = 8)
        self.R13 = ResBlock(in_channels = 448, out_channels = 256)
        self.T10 = SpatialTransformer(in_channels = 256, num_heads = 8)
        self.U1 = ResBlock(in_channels = 256, out_channels = 256, mode = "up")
        self.R14 = ResBlock(in_channels = 448, out_channels = 192)
        self.T11 = SpatialTransformer(in_channels = 192, num_heads = 6)
        self.R15 = ResBlock(in_channels = 384, out_channels = 192)
        self.T12 = SpatialTransformer(in_channels = 192, num_heads = 6)
        self.R16 = ResBlock(in_channels = 320, out_channels = 192)
        self.T13 = SpatialTransformer(in_channels = 192, num_heads = 6)
        self.U2 = ResBlock(in_channels = 192, out_channels = 192, mode = "up")
        self.R17 = ResBlock(in_channels = 320, out_channels = 128)
        self.T14 = SpatialTransformer(in_channels = 128, num_heads = 4)
        self.R18 = ResBlock(in_channels = 256, out_channels = 128)
        self.T15 = SpatialTransformer(in_channels = 128, num_heads = 4)
        self.R19 = ResBlock(in_channels = 192, out_channels = 128)
        self.T16 = SpatialTransformer(in_channels = 128, num_heads = 4)
        self.U3 = ResBlock(in_channels = 128, out_channels = 128, mode = "up")
        self.R20 = ResBlock(in_channels = 192, out_channels = 64)
        self.R21 = ResBlock(in_channels = 128, out_channels = 64)
        self.R22 = ResBlock(in_channels = 128, out_channels = 64)
        self.norm = nn.GroupNorm(num_groups = 32, num_channels = 64)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, padding = 1)


    def forward(self, x, timestep_emb, context):
        conv1_out = self.conv1(x)
        r1_out = self.R1(conv1_out, timestep_emb)
        r2_out = self.R2(r1_out, timestep_emb)
        d1_out = self.D1(r2_out, timestep_emb)
        r3_out = self.R3(d1_out, timestep_emb)
        t1_out = self.T1(r3_out, context)
        r4_out = self.R4(t1_out, timestep_emb)
        t2_out = self.T2(r4_out, context)
        d2_out = self.D2(t2_out, timestep_emb)
        r5_out = self.R5(d2_out, timestep_emb)
        t3_out = self.T3(r5_out, context)
        r6_out = self.R6(t3_out, timestep_emb)
        t4_out = self.T4(r6_out, context)
        d3_out = self.D3(t4_out, timestep_emb)
        r7_out = self.R7(d3_out, timestep_emb)
        t5_out = self.T5(r7_out, context)
        r8_out = self.R8(t5_out, timestep_emb)
        t6_out = self.T6(r8_out, context)

        r9_out = self.R9(t6_out, timestep_emb)
        t7_out = self.T7(r9_out, context)
        r10_out = self.R10(t7_out, timestep_emb)

        r11_out = self.R11(torch.cat((t6_out, r10_out), dim = 1), timestep_emb)
        t8_out = self.T8(r11_out, context)
        r12_out = self.R12(torch.cat((t5_out, t8_out), dim = 1), timestep_emb)
        t9_out = self.T9(r12_out, context)
        r13_out = self.R13(torch.cat((d3_out, t9_out), dim = 1), timestep_emb)
        t10_out = self.T10(r13_out, context)
        u1_out = self.U1(t10_out, timestep_emb)
        r14_out = self.R14(torch.cat((t4_out, u1_out), dim = 1), timestep_emb)
        t11_out =self.T11(r14_out, context)
        r15_out = self.R15(torch.cat((t3_out, t11_out), dim = 1), timestep_emb)
        t12_out = self.T12(r15_out, context)
        r16_out = self.R16(torch.cat((d2_out, t12_out), dim = 1), timestep_emb)
        t13_out = self.T13(r16_out, context)
        u2_out = self.U2(t13_out, timestep_emb)
        r17_out = self.R17(torch.cat((t2_out, u2_out), dim = 1), timestep_emb)
        t14_out = self.T14(r17_out, context)
        r18_out = self.R18(torch.cat((t1_out, t14_out), dim = 1), timestep_emb)
        t15_out = self.T15(r18_out, context)
        r19_out = self.R19(torch.cat((d1_out, t15_out), dim = 1), timestep_emb)
        t16_out = self.T16(r19_out, context)
        u3_out = self.U3(t16_out, timestep_emb)
        r20_out = self.R20(torch.cat((r2_out, u3_out), dim = 1), timestep_emb)
        r21_out = self.R21(torch.cat((r1_out, r20_out), dim = 1), timestep_emb)
        r22_out = self.R22(torch.cat((conv1_out, r21_out), dim = 1), timestep_emb)
        norm_out = self.norm(r22_out)
        conv2_out = self.conv2(norm_out)

        return conv2_out


