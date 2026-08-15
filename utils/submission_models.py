from utils.submission_imports import *


class AudioPool(nn.Module):
    """
    Pools along the time dimension using both Mean and Max.
    Preserves spatial/magnitude acoustic cues while preventing validation overfitting.
    """
    def forward(self, x):
        # x shape: [Batch, Channels, Time]
        mean_pool = torch.mean(x, dim=-1)
        max_pool = torch.max(x, dim=-1)[0]  # [0] extracts the actual maximum values tensor
        return torch.cat([mean_pool, max_pool], dim=1)  # Shape: [Batch, Channels * 2]

class DefaultCNN(nn.Module):
    def __init__(self, depth_no_cn_lay: int, width_no_kern: list, stride: list, kernel_size: list, sum: bool = False, input_length: int = 4800):
        super().__init__()
        self.depth = depth_no_cn_lay
        self.width = width_no_kern
        self.sum = sum
        
        # 1. Initialize the sequential container
        self.n = nn.Sequential()
        
        # 2.1 Loop through and build the convolutional layers
        for layer in range(self.depth):
            is_first = (layer == 0)
            
            # 2.2 Dynamic padding calculation based on dilation factor
            dilation_factor = 1 if is_first else 2
            current_kernel = kernel_size[layer]
            padding_val = (current_kernel - 1) * dilation_factor // 2
            
            self.n.append(nn.Conv1d(
                in_channels=width_no_kern[layer], 
                out_channels=width_no_kern[layer+1],
                kernel_size=current_kernel,
                stride=stride[layer], 
                padding=padding_val,
                dilation=dilation_factor,
                # Groups=4 on layer 0 keeps the 4 microphone channels isolated to preserve early TDOA phase cues
                groups=width_no_kern[0] if is_first else 1 
            ))
            self.n.append(nn.BatchNorm1d(width_no_kern[layer+1]))
            self.n.append(nn.LeakyReLU(0.01))

        # Append the advanced multi-pooling block
        self.n.append(AudioPool())
        
        # Calculate the exact linear input size from the final convolutional width
        flattened_features = width_no_kern[-1] * 2

        # Fully connected projection layers to map features to [X, Y] coordinates
        self.fc1 = nn.Linear(flattened_features, 32)
        self.activ = nn.LeakyReLU(negative_slope=0.01)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 2)
        
        if self.sum:      
            print("Printing summary... ", self.n)

    def forward(self, x):
        x = self.n(x)
        x = self.fc1(x)
        x = self.activ(x) 
        x = self.drop1(x)
        x = self.fc2(x) 
        return x
    


class AudioPool(nn.Module):
    """
    Pools along the time dimension using both Mean and Max.
    Preserves spatial/magnitude acoustic cues while preventing validation overfitting.
    """
    def forward(self, x):
        # x shape: [Batch, Channels, Time]
        mean_pool = torch.mean(x, dim=-1)
        max_pool = torch.max(x, dim=-1)[0]  # [0] extracts the actual maximum values tensor
        return torch.cat([mean_pool, max_pool], dim=1)  # Shape: [Batch, Channels * 2]

class NSourcesCNN(nn.Module):
    def __init__(self, depth_no_cn_lay: int, width_no_kern: list, stride: list, kernel_size: list, sum: bool = False, input_length: int = 4800):
        super().__init__()
        self.depth = depth_no_cn_lay
        self.width = width_no_kern
        self.sum = sum
        
        # 1. Initialize the sequential container
        self.n = nn.Sequential()
        
        # 2.1 Loop through and build the convolutional layers
        for layer in range(self.depth):
            is_first = (layer == 0)
            
            # 2.2 Dynamic padding calculation based on dilation factor
            dilation_factor = 1 if is_first else 2
            current_kernel = kernel_size[layer]
            padding_val = (current_kernel - 1) * dilation_factor // 2
            
            self.n.append(nn.Conv1d(
                in_channels=width_no_kern[layer], 
                out_channels=width_no_kern[layer+1],
                kernel_size=current_kernel,
                stride=stride[layer], 
                padding=padding_val,
                dilation=dilation_factor,
                # Groups=4 on layer 0 keeps the 4 microphone channels isolated to preserve early TDOA phase cues
                groups=width_no_kern[0] if is_first else 1 
            ))
            self.n.append(nn.BatchNorm1d(width_no_kern[layer+1]))
            self.n.append(nn.LeakyReLU(0.01))

        # Append the advanced multi-pooling block
        self.n.append(AudioPool())
        
        # Calculate the exact linear input size from the final convolutional width
        flattened_features = width_no_kern[-1] * 2

        # Fully connected projection layers to map features to [X, Y] coordinates
        self.fc1 = nn.Linear(flattened_features, 32)
        self.activ = nn.LeakyReLU(negative_slope=0.01)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 7)
        
        if self.sum:      
            print("Printing summary... ", self.n)

    def forward(self, x):
        x = self.n(x)
        x = self.fc1(x)
        x = self.activ(x) 
        x = self.drop1(x)
        x = self.fc2(x) 
        return x