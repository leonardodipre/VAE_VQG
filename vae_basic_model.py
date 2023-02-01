import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, z_dim=20):
        super().__init__()

        #encoder
        self.img_2hidden = nn.Linear(input_dim, hidden_dim)

        self.hid_2mu = nn.Linear(hidden_dim, z_dim)
        self.hid_2sigma = nn.Linear(hidden_dim, z_dim)

        self.relu = nn.ReLU()

    

    
    def encode(self, x):
        # q_phiz(z|x)
        h =self.relu( self.img_2hidden(x)  )
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)

        return mu , sigma

    

    def forward(self, x):
        mu, sigma = self.encode(x)

        epsilon = torch.rand_like(sigma)
        z_reparametrized = mu + sigma*epsilon

        

        #mu and sigma return for Loss function
        return z_reparametrized, mu, sigma



if __name__ == "__main__":

    x = torch.rand(1, 224)
    model = VAE(input_dim=224)

    x_reconstructed , mu , sigma = model(x)

    print("X_reconstruct")
    print(x_reconstructed)
    print(x_reconstructed.shape)
    print("MU")
    print(mu)
    print(mu.shape)
    print("Sigma")
    print(sigma)
    print(sigma.shape)