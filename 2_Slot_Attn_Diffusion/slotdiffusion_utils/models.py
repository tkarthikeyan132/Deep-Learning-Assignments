import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNet(nn.Module):
    '''
        Input dim: [B, 3, 128, 128]
        Output dim: [B, 1024, 192]
    '''
    def __init__(self, n=1):
        super().__init__()
        
        #1
        self.n = n

        self.module_dict = nn.ModuleDict()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        #2 and #3
        
        for i in range(self.n):
            self.module_dict.update({
                f"line1-{i}": nn.Sequential(
                    nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1, stride = 1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1, stride = 1),
                    nn.BatchNorm2d(16)
                )
            })
        
        #4 and #5
        self.dash2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(32)
        )
        
        for i in range(self.n - 1):
            self.module_dict.update({
                f"line2-{i}": nn.Sequential(
                    nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, stride = 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, stride = 1),
                    nn.BatchNorm2d(32)
            )
            })


        #6 and #7
        self.dash3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64)
        )

        for i in range(self.n - 1):
            self.module_dict.update({
                f"line3-{i}": nn.Sequential(
                    nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
                    nn.BatchNorm2d(64)
            )
            })

        #res2
        self.residual_connection2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, stride = 2),
            nn.BatchNorm2d(32)
        )

        #res3
        self.residual_connection3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, stride = 2),
            nn.BatchNorm2d(64)
        )

        self.head = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(192)
        )

        self.positional_embedding = nn.Linear(in_features = 4, out_features = 192)
        
    def forward(self, x):
        #1
        x = self.initial_conv(x)

        #SET 1
        for i in range(self.n):
            identity = x
            x = self.module_dict[f"line1-{i}"](x)
            x = F.relu(x + identity)
        
        #SET 2
        identity = x
        x = self.dash2(x)
        x = F.relu(x + self.residual_connection2(identity))

        for i in range(self.n - 1):
            identity = x
            x = self.module_dict[f"line2-{i}"](x)
            x = F.relu(x + identity)

        #SET 3
        identity = x
        x = self.dash3(x)
        x = F.relu(x + self.residual_connection3(identity))
        for i in range(self.n - 1):
            identity = x
            x = self.module_dict[f"line3-{i}"](x)
            x = F.relu(x + identity)

        #head
        x = self.head(x)

        x = x.permute(0, 2, 3, 1) # x is (B, 32, 32, 64)

        pos_grid = self.position_grid() # pos_grid is (1, 32, 32, 4)
        pos_grid = self.positional_embedding(pos_grid) # pos_grid is (1, 32, 32, 64)
    
        x = x + pos_grid # x is (B, 32, 32, 64)
        
        x = torch.flatten(x, start_dim = 1, end_dim = 2) # x is (B, 1024, 64)

        return x

    def position_grid(self):
        DIM = 32

        x = np.linspace(0.0, 1.0, DIM) # x is list of equally spaced entries
        y = np.linspace(0.0, 1.0, DIM) # y is list of equally spaced entries

        nx, ny = np.meshgrid(x, y, indexing="ij") # nx and ny are DIM X DIM row stacked and column stacked

        grid = np.stack((nx, ny), axis = -1) # grid is (DIM, DIM, 2)
        grid = np.expand_dims(grid, axis = 0) # similar to unsqueeze(0) .. post which grid is (1, DIM, DIM, 2)
        grid = grid.astype(np.float32)
        grid = np.concatenate([grid, 1.0 - grid], axis = -1) # creates a 4 dimensional position for each of the pixel in (DIM, DIM) .. post which grid is (1, DIM, DIM, 4)
        
        return torch.from_numpy(grid).to(device)


class CNN(nn.Module):
    '''
        Input dim: [B, 3, 128, 128]
        Output dim: [B, 1024, 64]
    '''
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5, padding = 2, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, padding = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, padding = 2, stride = 2)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, padding = 2, stride = 1)

        self.positional_embedding = nn.Linear(in_features = 4, out_features = 64)

    def forward(self, x): # x is (B, 3, 128, 128)
        x = F.relu(self.conv1(x)) # x is (B, 64, 128, 128)
        x = F.relu(self.conv2(x)) # x is (B, 64, 64, 64)
        x = F.relu(self.conv3(x)) # x is (B, 64, 32, 32)
        x = F.relu(self.conv4(x)) # x is (B, 64, 32, 32)
        x = x.permute(0, 2, 3, 1) # x is (B, 32, 32, 64)

        pos_grid = self.position_grid() # pos_grid is (1, 32, 32, 4)
        pos_grid = self.positional_embedding(pos_grid) # pos_grid is (1, 32, 32, 64)
    
        x = x + pos_grid # x is (B, 32, 32, 64)
        
        x = torch.flatten(x, start_dim = 1, end_dim = 2) # x is (B, 1024, 64)

        return x

    def position_grid(self):
        DIM = 32

        x = np.linspace(0.0, 1.0, DIM) # x is list of equally spaced entries
        y = np.linspace(0.0, 1.0, DIM) # y is list of equally spaced entries

        nx, ny = np.meshgrid(x, y, indexing="ij") # nx and ny are DIM X DIM row stacked and column stacked

        grid = np.stack((nx, ny), axis = -1) # grid is (DIM, DIM, 2)
        grid = np.expand_dims(grid, axis = 0) # similar to unsqueeze(0) .. post which grid is (1, DIM, DIM, 2)
        grid = grid.astype(np.float32)
        grid = np.concatenate([grid, 1.0 - grid], axis = -1) # creates a 4 dimensional position for each of the pixel in (DIM, DIM) .. post which grid is (1, DIM, DIM, 4)
        
        return torch.from_numpy(grid)


class SlotAttention(nn.Module):
    '''
        Input dim: [B, N, DIM]
        Output dim: [B, K, DIM]
    '''
    def __init__(self, N_SLOTS=11, DIM=192):
        super().__init__()
        
        HIDDEN_DIM = 128 #This hidden_dimension is for MLP
        EPSILON = 1e-8
        ITERATIONS = 3

        self.n_slots = N_SLOTS
        self.iterations = ITERATIONS
        self.eps = EPSILON
        self.scale = DIM ** -0.5
        self.dim = DIM

        self.slots_mu = nn.Parameter(torch.randn(1, 1, DIM))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, DIM))

        self.Q = nn.Linear(DIM, DIM)
        self.K = nn.Linear(DIM, DIM)
        self.V = nn.Linear(DIM, DIM)

        self.gru = nn.GRUCell(DIM, DIM)  # GRU cell being used because hidden state for each element in the batch have to be passed

        self.mlp = nn.Sequential(
            nn.Linear(DIM, HIDDEN_DIM), # (B, K, HID_DIM)
            nn.ReLU(), # (B, K, HID_DIM)
            nn.Linear(HIDDEN_DIM, DIM) # (B, K, DIM)
        )

        self.layer_norm_in  = nn.LayerNorm(DIM)
        self.layer_norm_slots  = nn.LayerNorm(DIM)
        self.layer_norm_pre_mlp = nn.LayerNorm(DIM)

    def forward(self, inputs, n_slots = 11):
        B, _, _ = inputs.shape
        n_slots = n_slots if n_slots is not None else self.n_slots
        
        mu = self.slots_mu.expand(B, n_slots, -1) # (B, K, DIM)
        sigma = self.slots_sigma.expand(B, n_slots, -1) # (B, K, DIM)
        
        slots = torch.normal(mu, sigma) # (B, K, DIM) .. sample from the normal distribution elementwise

        inputs = self.layer_norm_in(inputs) # (B, N, DIM)
                
        key_vec, val_vec = self.K(inputs), self.V(inputs) # both key and val are of dimension (B, N, DIM)

        for _ in range(self.iterations):
            slots_prev = slots

            slots = self.layer_norm_slots(slots) # (B, K, DIM)
            
            query_vec = self.Q(slots) # (B, K, DIM)

            dot_result = torch.einsum('bkd,bnd->bkn', query_vec, key_vec) * self.scale # (B, K, N) .... query is (B, K, DIM) and key is (B, N, DIM) and it is converted to (B, K, N) #Einstein summation convention
            
            attn = dot_result.softmax(dim=1) + self.eps  # (B, K, N)
            attn_output = attn
            
            attn = attn / attn.sum(dim=-1, keepdim=True) # (B, K, N) .. for the weighted mean
            
            updates = torch.einsum('bnd,bkn->bkd', val_vec, attn) # (B, K, DIM) .... val is (B, N, DIM) and attn is (B, K, N) and it is converted to (B, K, DIM) #Einstein summation convention
            
            slots = self.gru(
                updates.reshape(-1, self.dim),
                slots_prev.reshape(-1, self.dim)
            ) # (B*K, DIM)
            
            slots = slots.reshape(B, -1, self.dim) # (B, K, DIM)

            slots_mlp = self.layer_norm_pre_mlp(slots) # (B, K, DIM)
            
            slots_mlp = self.mlp(slots_mlp) # (B, K, DIM)

            slots = slots + slots_mlp # (B, K, DIM)

        return slots, attn_output


class DeCNN(nn.Module):
    '''
        Input dim: [B, K, 64]
        Output dim: [B, K, 128, 128, 4]
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(in_channels = 64, out_channels = 4, kernel_size = 3, stride=(1, 1), padding=1)
        
        self.positional_embedding = nn.Linear(in_features = 4, out_features = 64)

    def forward(self, x):
        B, _, _ = x.shape
        #Spatial broadcasting
        x = x.reshape((-1, x.shape[-1])).unsqueeze(1).unsqueeze(2) # (B*K, 1, 1, 64)
        x = x.repeat((1, 8, 8, 1)) # (B*K, 8, 8, 64)

        pos_grid = self.position_grid() # pos_grid is (1, 8, 8, 4)
        pos_grid = self.positional_embedding(pos_grid) # pos_grid is (1, 8, 8, 64)
    
        x = x + pos_grid # x is (B*K, 8, 8, 64)
            
        x = x.permute(0,3,1,2) # x is (B*K, 64, 8, 8)
        
        x = F.relu(self.conv1(x)) # x is (B*K, 64, 16, 16)
        x = F.relu(self.conv2(x)) # x is (B*K, 64, 32, 32)
        x = F.relu(self.conv3(x)) # x is (B*K, 64, 64, 64)
        x = F.relu(self.conv4(x)) # x is (B*K, 64, 128, 128)
        x = F.relu(self.conv5(x)) # x is (B*K, 64, 128, 128)
        x = self.conv6(x) # x is (B*K, 4, 128, 128)
        
        x = x.permute(0,2,3,1) # x is (B*K, 128, 128, 4)
        x = x.reshape(B, -1, x.shape[1], x.shape[2], x.shape[3]) # x is (B, K, 128, 128, 4)
        return x

    def position_grid(self):
        DIM = 8

        x = np.linspace(0.0, 1.0, DIM) # x is list of equally spaced entries
        y = np.linspace(0.0, 1.0, DIM) # y is list of equally spaced entries

        nx, ny = np.meshgrid(x, y, indexing="ij") # nx and ny are DIM X DIM row stacked and column stacked

        grid = np.stack((nx, ny), axis = -1) # grid is (DIM, DIM, 2)
        grid = np.expand_dims(grid, axis = 0) # similar to unsqueeze(0) .. post which grid is (1, DIM, DIM, 2)
        grid = grid.astype(np.float32)
        grid = np.concatenate([grid, 1.0 - grid], axis = -1) # creates a 4 dimensional position for each of the pixel in (DIM, DIM) .. post which grid is (1, DIM, DIM, 4)
        
        return torch.from_numpy(grid).to(device)


class Encoder(nn.Module):
    '''
        Input dim: [B, 3, 128, 128]
        Output dim: [B, 11, 192]
    '''
    def __init__(self):
        super().__init__()
        self.Resnet = ResNet()
        self.SlotAttn = SlotAttention()
        
    def forward(self, x):
        x = self.Resnet(x) # (B, 1024, 64)
        slots, attn_output = self.SlotAttn(x) # (B, K, 64)
        
        return slots, attn_output


class Decoder(nn.Module):
    '''
        Input dim: [B, 11, 192]
        Output dim: [B, 3, 128, 128]
    '''
    def __init__(self):
        super().__init__()
        self.DeCNN = DeCNN()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, slots):
        y = self.DeCNN(slots) # (B, K, 128, 128, 4)

        re_constructed_images, masks = y.split([3,1], dim=-1) # (B, K, 128, 128, 3) and (B, K, 128, 128, 1)
        masks = self.softmax(masks) # To normalize across slots

        re_constructed_image = torch.sum(re_constructed_images * masks, dim=1)  # (B, 128, 128, 3)

        re_constructed_image = re_constructed_image.permute(0,3,1,2) # (B, 3, 128, 128)
        return re_constructed_image, re_constructed_images, masks