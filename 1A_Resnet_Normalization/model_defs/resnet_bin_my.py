import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchInstanceNorm_My(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine = True, momentum = 0.1, rho = 0.5, device = "cuda"):
        super(BatchInstanceNorm_My, self).__init__()
        self.num_features = num_features
        self.eps = torch.tensor(eps) #No shape
        self.affine = affine
        self.num_features = num_features
        self.momentum = momentum
        self.rho = nn.Parameter(torch.tensor(rho))

        # Initialize parameters
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features)) #num_features
            self.beta = nn.Parameter(torch.zeros(num_features)) #num_features

        # self.running_mean = torch.zeros(num_features, device = device) #num_features
        # self.running_var = torch.ones(num_features, device = device) #num_features
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            batch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True) #num_features
            batch_var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True) #num_features

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze() #num_features
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze() #num_features
        
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        norm_x_batch = (x - batch_mean.reshape(1,-1,1,1)) / torch.sqrt(batch_var.reshape(1,-1,1,1) + self.eps) #N x C x H x W

        
        mean_instance = torch.mean(x, dim=(2, 3), keepdim=True) #batch x num_features
        var_instance = torch.var(x, dim=(2, 3), unbiased=False, keepdim=True) #batch x num_features

        # Normalize input
        norm_x_instance = (x - mean_instance) / torch.sqrt(var_instance + self.eps) #N x C x H x W

        normalized_x = self.rho * norm_x_batch + (1 - self.rho) * norm_x_instance 
        
        if self.affine:
            # Scale and shift
            normalized_x = normalized_x * self.gamma.reshape(1,-1,1,1) + self.beta.reshape(1,-1,1,1) #N x C x H x W

        return normalized_x #N x C x H x W

class ResNet(nn.Module):
    def __init__(self, n , r ):
        super().__init__()
        
        #1
        self.n = n
        self.r = r
        self.module_dict = nn.ModuleDict()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1, stride = 1),
            BatchInstanceNorm_My(16),
            nn.ReLU()
        )

        #2 and #3
        
        for i in range(self.n):
            self.module_dict.update({
                f"line1-{i}": nn.Sequential(
                    nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1, stride = 1),
                    BatchInstanceNorm_My(16),
                    nn.ReLU(),
                    nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1, stride = 1),
                    BatchInstanceNorm_My(16)
                )
            })
        
        #4 and #5
        self.dash2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, stride = 2),
            BatchInstanceNorm_My(32),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, stride = 1),
            BatchInstanceNorm_My(32)
        )
        
        for i in range(self.n - 1):
            self.module_dict.update({
                f"line2-{i}": nn.Sequential(
                    nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, stride = 1),
                    BatchInstanceNorm_My(32),
                    nn.ReLU(),
                    nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1, stride = 1),
                    BatchInstanceNorm_My(32)
            )
            })


        #6 and #7
        self.dash3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, stride = 2),
            BatchInstanceNorm_My(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
            BatchInstanceNorm_My(64)
        )

        for i in range(self.n - 1):
            self.module_dict.update({
                f"line3-{i}": nn.Sequential(
                    nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
                    BatchInstanceNorm_My(64),
                    nn.ReLU(),
                    nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
                    BatchInstanceNorm_My(64)
            )
            })
        
        # self.avg = nn.AvgPool2d(kernel_size = 2, stride = 2)
        # self.linear = nn.Linear(in_features = 64*32*32, out_features = self.r)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_features = 64, out_features = self.r)

        #res2
        self.residual_connection2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, stride = 2),
            BatchInstanceNorm_My(32)
        )

        #res3
        self.residual_connection3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, stride = 2),
            BatchInstanceNorm_My(64)
        )

        
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
        
        x = self.avg(x)
        x = x.flatten(start_dim = 1)
        x = self.linear(x)
        
        return x