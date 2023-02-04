import pandas as pd
import torch
from torch import nn, optim
from  model  import CNNtoRNN
from dataloader import get_loader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
import json
from eval import eval1
import numpy as np
torch.manual_seed(258713695246)


def dropout_and_replace(tensor_batch, dropout_rate, replace_token, exclude_list):
   
    result_batch = []
    for tensor in tensor_batch:
        mask = np.random.binomial(1, 1 - dropout_rate, tensor.shape).astype(bool)
        tensor = np.where(np.logical_or(mask, np.isin(tensor, exclude_list)), tensor, replace_token)
        result_batch.append(tensor)

    return  torch.tensor(result_batch)
"""

def dropout_and_replace(tensor_batch, dropout_rate, replace_token, exclude_list):
    result_batch = []
    exclude_list = torch.tensor(exclude_list, dtype=torch.long, device=tensor_batch[0].device)
    for tensor in tensor_batch:
        mask = (torch.rand(tensor.shape, device=tensor.device) > dropout_rate).bool()
        tensor = torch.where(mask | tensor.unsqueeze(-1).eq(exclude_list).any(-1), tensor, replace_token)
        result_batch.append(tensor)
    combined = torch.cat(result_batch)
    return combined
"""

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.2):

    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epoch):
            L[int(i+c*period)] = v
           
            v += step
            i += 1
    return L  




def train():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Declare transformations (later)
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            # transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )

    ############################################################################################################################################
                                                        #DIRECTORY#
    
    
    csv = r'train_coco.csv'
    imm_dir =r'C:\Users\leona\Desktop\COCO-dataset\coco-2014\train2014\train2014'


    freq_threshold = 4 # 4019 vocab

    ############################################################################################################################################


    loader, dataset = get_loader(
        csv, imm_dir, freq_threshold, transform=transform , num_workers=2,
        )
  

    # Hyperparameters
    embed_size = 224
    hidden_size = 224
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 1e-4
    num_epochs = 20
    train_CNN = False
    ###########################################################################

    # Model declaration
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    # Criterion declaration
    weight = torch.ones(vocab_size).to(device)
    weight[0] = 0 # Ignore the padding
    weight[3] = 0 # Ignore the unk token
    
    # Optimizer declaration
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #new Losss######################################
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    ###############################################

    for name, param in model.encoderCNN.resNet.named_parameters():
        # Only finetune the CNN
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

   
   #calulate the list for B value for KL vanquisch
    beta_np_inc = frange_cycle_linear(0.0, 1.0, num_epochs, 1, 0.25)


    for epoch in range(num_epochs):
        
        PATH = "save_Model"
        torch.save(model.state_dict(), PATH)
        
        epoch_loss = 0

        step = 0 
        step_loss = 0 
       
        for  i, (imgs, questions, lengths, index) in  enumerate(tqdm(loader)):

            model.train()

            imgs = imgs.to(device)
                       

            #drop some question add <unk> tokken
            exclude_list = [0, 1, 2, 3, 4]
            

          
            questions = dropout_and_replace(questions, 0.5, 3, exclude_list)
            questions = questions.to(device)
            

            outputs, mu, sigma = model(imgs,questions,lengths)

            
            targets = pack_padded_sequence(questions[:, 1:], [l-1 for l in lengths], batch_first=True)[0]

            ######################### KL Divergent #########################
            kl_divergent = -torch.sum(1+torch.log(sigma.pow(2)) - mu.pow(2)-sigma.pow(2))
            
            #calcolo loss 2.0
            reconstruction_loss = loss_fn(outputs, targets) +  beta_np_inc[epoch]* kl_divergent
           

            ###########################################################################

                     
            epoch_loss += reconstruction_loss.item()
            step_loss += reconstruction_loss.item()

            optimizer.zero_grad()
            reconstruction_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            step+=1

           
            if(step%250 == 0 and step!=0):
                print(f"Step{step}/Epoch{epoch}], Loss: {step_loss/250:.4f}")
                step_loss = 0
                eval1(model, device, dataset, "prova.jpg")
                
                eval1(model, device, dataset, "prova2.jpg")
            
                

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(loader):.4f}")
       

if __name__ == "__main__":
    train()
