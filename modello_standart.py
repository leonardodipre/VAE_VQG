import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNencoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(CNNencoder,self).__init__()
        self.train_CNN = train_CNN
        self.resNet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.preprocess = ResNet18_Weights.DEFAULT.transforms()
        self.resNet.fc = nn.Linear(self.resNet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resNet(self.preprocess(images))
        return features


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = CNNencoder(embed_size)
        

    def forward(self, images):
        features = self.encoderCNN(images)
        
        return features

   

embed_size= 224
hidden_size = 224

model = CNNtoRNN(embed_size)

x = torch.rand(1,3,224,224)
print(model(x).shape)