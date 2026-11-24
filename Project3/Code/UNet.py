import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    #We make this Class as for all layers in the U-Net, except bottleneck, we perform a double convolution.
    def __init__(self, 
                 input_channels, 
                 in_channels, 
                 out_channels,
                 act, 
                 kernel_size=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2), 
            act(),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            act(),
        )

    def forward(self, x):
        return self.conv(x)


class MultiKernelBlock(nn.Module):
    def __init__(self,
                 input_channels,
                 out_channels,
                 act = nn.ReLU,
                 # Use only odd kernel sizes
                 kernel_sizes = (1, 3, 5)):
        super().__init__()

        self.branches = nn.ModuleList([nn.Conv1d(input_channels, input_channels, kernel_size=k, padding=k//2) for k in kernel_sizes])

        concat_channels = input_channels * len(kernel_sizes)

        # Normalise + activate the concatenated features, then mix with a 1x1 conv
        self.post_concat = nn.Sequential(
            nn.BatchNorm1d(concat_channels),
            act(),
        )
        self.mix = nn.Sequential(
            nn.Conv1d(concat_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            act(),
        )

    def forward(self, x):
        # x: (batch, channels, length)
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        out = self.post_concat(out)
        return self.mix(out)

    
class UNet(nn.Module):
    """
    A UNet architexture that is four layers deep and can have different activation
    functions in the encoder and decoder part.

    Attributes
    ----------
    encoder: nn.ModuleList
        All convolutions in the encoder part
    decoder: nn.ModuleList
        All convolutions in the decoder part
    pool: nn.Module 
        Pooling function
    final_conv: nn.Module 
        Final convolution to get the output
    
    Methods
    -------
    forward(x: torch.Tensor)
        Processes the images and returns the output
    """
    def __init__(self, 
                 input_channels: int, 
                 in_channels: int, 
                 out_channels: int,
                 pooling: type[nn.Module],
                 activationenc: list[callable],
                 activationdec: list[callable], 
                 features: tuple[int] = (16, 32, 64),
                 MK_kernel_sizes = (1, 3, 5),
                 # stem kernel, up kernel, decoder kernel
                 kernel_sizes = (7, 3, 5)):
        super().__init__()
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception 
        # of the last block which does not include a max-pooling layer.
        # -------
        # input: whatever you want
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = pooling(kernel_size=2, stride=2)
        
        self.stem_conv = nn.Conv1d(input_channels, in_channels, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)

        #Making Arcitecture for Encoding and Bottleneck
        for i, feature in enumerate(features):
            self.encoder.append(MultiKernelBlock(in_channels, 
                                                feature, 
                                                activationenc[i],
                                                MK_kernel_sizes)
                                                )
            
            in_channels = feature


        # --||-- Decoder
        for i in range(1, len(features)):
            self.decoder.append(
                nn.ConvTranspose1d(features[-i], features[-i], kernel_size=2, stride=2)
            )
            self.decoder.append(
                DoubleConv(features[-(i + 1)] + features[-i], features[-(i + 1)], features[-(i + 1)], activationdec[i-1], kernel_size=kernel_sizes[2])
            )

        #Final convolution to get results
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        skip_connections = []

        x = self.stem_conv(x)

        #Going through the encoder and bottleneck
        for down in self.encoder[:-1]:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
 
        #Bottleneck!
        x = self.encoder[-1](x)

        #To reverse connections
        skip_connections = skip_connections[::-1]
 
        # Changed from range(0, len(self.decoder) - 1) to  range(0, len(self.decoder), 2)
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            # And from [idx] to [idx // 2]
            skip_connection = skip_connections[idx // 2]
 
            if x.shape[2:] != skip_connection.shape[2:]:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode='linear', align_corners=False
                )
            
            #Overlapping the images
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
 
        return self.final_conv(x)



# Check main
if __name__ == '__main__':
     # This code won't run if this file is imported.
    x = torch.rand((2,1,1024))
    out = 4
    test = UNet(1, 16, out, nn.MaxPool1d, [nn.ReLU]*3, [nn.ReLU]*2)
    print(test.forward(x).shape)
        
