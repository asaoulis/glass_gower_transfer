
import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_block=False):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        if not first_block:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True, padding_mode='circular'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_block=False):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        if not first_block:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='circular'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True, padding_mode='circular'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

#####################################################################################
class model_o3_err(nn.Module):
    def __init__(self, num_outputs, hidden, dr=0.35, channels=1, predict_sigmas = False):
        super(model_o3_err, self).__init__()
        self.predict_sigmas = predict_sigmas
        self.num_outputs = num_outputs
        if predict_sigmas:
            num_outputs =  2* self.num_outputs
        
        # input: 1x256x256 ---------------> output: 2*hiddenx128x128
        self.C01 = nn.Conv2d(channels,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=2, stride=2, padding=0, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(2*hidden)
        self.B02 = nn.BatchNorm2d(2*hidden)
        self.B03 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
        self.C11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(4*hidden)
        self.B12 = nn.BatchNorm2d(4*hidden)
        self.B13 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(8*hidden)
        self.B22 = nn.BatchNorm2d(8*hidden)
        self.B23 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C31 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(16*hidden)
        self.B32 = nn.BatchNorm2d(16*hidden)
        self.B33 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(32*hidden)
        self.B42 = nn.BatchNorm2d(32*hidden)
        self.B43 = nn.BatchNorm2d(32*hidden)
        
        # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
        self.C51 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(64*hidden)
        self.B52 = nn.BatchNorm2d(64*hidden)
        self.B53 = nn.BatchNorm2d(64*hidden)

        # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.C61 = nn.Conv2d(64*hidden, 128*hidden, kernel_size=4, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        # self.B61 = nn.BatchNorm2d(128*hidden)
        self.B61 = nn.Identity() # torch doesn't like BN on 1dim data

        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1  = nn.Linear(128*hidden, 64*hidden)  
        self.FC2  = nn.Linear(64*hidden,  num_outputs)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):

        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))

        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))

        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))

        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))

        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))

        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))

        x = self.LeakyReLU(self.B61(self.C61(x)))

        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)
        if self.predict_sigmas:
            # enforce the errors to be positive
            y = torch.clone(x)
            y[:,self.num_outputs:2*self.num_outputs] = torch.square(x[:,self.num_outputs:2*self.num_outputs])

            return y
        
        return x
####################################################################################
####################################################################################

# Example usage


def build_resnet(num_outputs, pretrained=True):
    
    resnet = models.resnet18(pretrained=pretrained)

    # Copy weights from the original layer
    original_weights = resnet.conv1.weight.data

    # Average the weights across the RGB channels
    new_weights = original_weights.mean(dim=1, keepdim=True)

    # Replace the conv1 layer and assign the new weights
    resnet.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=resnet.conv1.out_channels,
        kernel_size=resnet.conv1.kernel_size,
        stride=resnet.conv1.stride,
        padding=resnet.conv1.padding,
        bias=resnet.conv1.bias is not None
    )
    resnet.conv1.weight.data = new_weights

    # add two fc layers
    resnet.fc = nn.Sequential(
        # nn.Linear(512, 256),
        # nn.ReLU(),
        nn.Linear(512, num_outputs),

    )
    return resnet

def build_convnext(num_outputs, pretrained=True):
    convnext = models.convnext_tiny(pretrained=pretrained)
    
    # Get the original first convolution layer
    original_conv = convnext.features[0][0]  # First conv layer in ConvNeXt
    original_weights = original_conv.weight.data
    
    # Average the weights across RGB channels
    new_weights = original_weights.mean(dim=1, keepdim=True)
    
    # Replace first conv layer with single-channel input
    convnext.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )
    convnext.features[0][0].weight.data = new_weights
    
    # Modify classifier head
    in_features = convnext.classifier[2].in_features
    convnext.classifier = nn.Sequential(
        nn.Flatten(),
        # nn.Linear(in_features, 256),
        # nn.ReLU(),
        
        nn.Linear(in_features, num_outputs),
    )
    
    return convnext



_MODEL_BUILDERS = {
    "o3": lambda num_outputs, **kwargs: model_o3_err(num_outputs, hidden=12).to(device='cuda'),
    "resnet": lambda num_outputs, pretrained=True, **kwargs: build_resnet(num_outputs, pretrained=pretrained),
    "convnext": lambda num_outputs, pretrained=True, **kwargs: build_convnext(num_outputs, pretrained=pretrained)
}
