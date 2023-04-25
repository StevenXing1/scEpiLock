from torch import nn
import torch.nn.functional as F
import torch
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(ResBlock, self).__init__()
        self.skip = nn.Sequential()

        if in_channel != out_channel:
          self.skip = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size-1, padding = kernel_size//2-1),
            nn.BatchNorm1d(out_channel))
        else:
          self.skip = None

        self.res = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, padding = kernel_size//2-1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channel, out_channel, kernel_size, padding = kernel_size//2),
            nn.BatchNorm1d(out_channel)
        )
        #self.shortcut = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size), nn.BatchNorm1d(out_channel))
            
    def forward(self, x):
        identity = x
        output = self.res(x)

        if self.skip is not None:
            identity = self.skip(x)

        output += identity
        output = F.relu(output)
        return output



class scEpiLock(nn.Module):
    def __init__(self, n_class):
        super(scEpiLock, self).__init__()
        self.Conv1 = nn.Sequential(nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8), nn.BatchNorm1d(320), nn.ReLU())
        #self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        #self.bn2 = nn.BatchNorm1d(out_channels = 320)
        
        #self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8) #padding
        self.Conv2 = nn.Sequential(ResBlock(320, 480, 8))
       # self.Conv2 = self.make_layer(ResBlock(320, 480, 8), 2, stride=6)

        #self.Conv3 = nn.Conv1d(in_channels=480, out_channels=480, kernel_size=4)
        self.Conv3 = nn.Sequential(ResBlock(480, 480, 4))
        #self.bn3 = nn.BatchNorm1d(out_channels = 480)
        
        #self.Conv3 = self.make_layer(ResBlock(480, 480, 4), 2, stride=6)
        #self.Conv4 = self.make_layer(ResBlock(480, 480, 4), 2, stride=6)
        self.Conv4 = nn.Sequential(ResBlock(480, 480, 4))
        self.Conv5 = nn.Sequential(ResBlock(480, 480, 4))
        self.Conv6 = nn.Sequential(ResBlock(480, 480, 4))
        self.Conv7 = nn.Sequential(ResBlock(480, 480, 4))
        self.Conv8 = nn.Sequential(ResBlock(480, 480, 4))
        #self.Conv5 = self.make_layer(ResBlock(480, 480, 4), 2, stride=6)
        #self.Conv6 = self.make_layer(ResBlock(480, 480, 4), 2, stride=6)
        #self.Conv7 = self.make_layer(ResBlock(480, 480, 4), 2, stride=6)
        #self.Conv8 = self.make_layer(ResBlock(480, 480, 4), 2, stride=6)
        #self.Conv4 = nn.Conv1d(in_channels=480, out_channels=480, kernel_size=4)
        #self.bn4 = nn.BatchNorm1d(out_channels = 480)

        #self.Conv5 = nn.Conv1d(in_channels=480, out_channels=480, kernel_size=4)
        #self.bn5 = nn.BatchNorm1d(out_channels = 480)

        #self.Conv6 = nn.Conv1d(in_channels=480, out_channels=480, kernel_size=4)
        #self.bn6 = nn.BatchNorm1d(out_channels = 480)

        #self.Conv7 = nn.Conv1d(in_channels=480, out_channels=480, kernel_size=4)
        #self.bn7 = nn.BatchNorm1d(out_channels = 480)

        #self.Conv8 = nn.Conv1d(in_channels=480, out_channels=480, kernel_size=4)
        #self.bn8 = nn.BatchNorm1d(out_channels = 480)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)

        self.Maxpool2 = nn.MaxPool1d(kernel_size=6, stride=6) #or make it 8(fewer training time)

        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)


        self.Linear1 = nn.Linear(74880, 925) #change to 10000 for transformer
        
        self.Linear2 = nn.Linear(925, 7) 

    #def make_layer(self, block, num_blocks, stride):
     #   strides = [stride] + [1] * (num_blocks - 1)
    #    layers = []
    #    for stride in strides:
    #        layers.append(block)
    #    return nn.Sequential(*layers)
    
    def forward(self, input):
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 4, 1000]
        # Output Tensor Shape: [batch_size, 320, 993]
        x = self.Conv1(input)
        #x = self.bn1(x)
        x = F.relu(x)
        # # Pooling Layer 1
        # # Input Tensor Shape: [batch_size, 320, 993]
        # # Output Tensor Shape: [batch_size, 320, 248]
        # x = self.Maxpool(x)
        # x = self.Drop1(x)

        # Convolution Layer 2
        # Input Tensor Shape: [batch_size, 320, 993]
        # Output Tensor Shape: [batch_size, 480, 986]
        x = self.Conv2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        # Pooling Layer 1320
        # Input Tensor Shape: [batch_size, 480, 986]
        # Output Tensor Shape: [batch_size, 480, 246]
        x = self.Drop1(x)

        # Convolution Layer 3
        # Input Tensor Shape: [batch_size, 480, 246]
        # Output Tensor Shape: [batch_size, 960, 243]
        x = self.Conv3(x)
        #x = self.bn3(x)
        x = F.relu(x)
        # Pooling Layer 2
        # Input Tensor Shape: [batch_size, 480, 243]
        # Output Tensor Shape: [batch_size, 480, 60]
        x = self.Maxpool(x)
        x = self.Drop2(x)

        # Convolution Layer 4
        # Input Tensor Shape: [batch_size, 960, 60]
        # Output Tensor Shape: [batch_size, 1024, 57]
        x = self.Conv4(x)
        #x = self.bn4(x)
        x = F.relu(x)

        x = self.Drop2(x)

        #Convolution layer 5
        x = self.Conv5(x)
        #x = self.bn5(x)
        x = F.relu(x)


        x = self.Drop2(x)

        #Convolution layer 6
        x = self.Conv6(x)
        #x = self.bn6(x)
        x = F.relu(x)

        x = self.Drop2(x)

        #Convolution layer 7
        x = self.Conv7(x)
        #x = self.bn7(x)
        x = self.Maxpool(x)
        x = F.relu(x)
        
        x = self.Drop2(x)

        #Convolution layer 8
        x = self.Conv8(x)
        #x = self.bn8(x)
        x = self.Maxpool(x)
        x = F.relu(x)

        x = self.Drop2(x)

        # Pooling Layer 3
        # Input Tensor Shape: [batch_size, 1024, 57]
        # Output Tensor Shape: [batch_size, 1024, 14]
        #x = self.Maxpool(x)
        
        x = x.view(-1, 74880)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x
    
#if __name__ == '__main__':
#    x = torch.rand(1, 4, 10000)
    
#    print(x)
#    model = scEpiLock(8)
#    model(x)
