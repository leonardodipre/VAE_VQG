import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
import numpy as np 
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg





spacy_eng =  spacy.load("en_core_web_sm")


#inserrisco manualmete il ? come carattere 4 da imparrae a fine di ogni domanda
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", 4:"?"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3, "?":4}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        list_numerized = []
        for token in tokenized_text:
            try: 
                list_numerized.append(self.stoi[token])
            except:
                list_numerized.append(self.stoi["<UNK>"])

        return list_numerized

class CocoDataset(Dataset):
    
    def __init__(self,  csv, immage_dir, freq_threshold, transform=None):
        
        #Path csv 
        self.csv = csv        # csv con immagini domande ecc        
        self.immage_dir = immage_dir         # cartella immagini coco
      
    

        #leggo i csv
        self.file_csv =pd.read_csv(csv)         # ID IMMAGINI
        
        "id,Domande,immagine_id"

        #salvo id e domande
        self.id_imm_list = self.file_csv["id"]
        self.questions_list = self.file_csv["domanda"]
        self.coco_link_list = self.file_csv["coco_id"]
       
        self.transform = transform


        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.questions_list.tolist())

    def __len__(self):
        return len(self.questions_list)

    def __getitem__(self, index):

       
        domanda_imm = self.questions_list[index] 
        id_imm      = self.id_imm_list[index]  
        coco_id_imm = self.coco_link_list[index]  

        img = Image.open(os.path.join(self.immage_dir, coco_id_imm)).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img) 

        numericalized_question = [self.vocab.stoi["<SOS>"]]
        numericalized_question += self.vocab.numericalize(domanda_imm)
        numericalized_question.append(self.vocab.stoi["<EOS>"])

       

        return img, torch.tensor(numericalized_question),  len(numericalized_question), index

    
            
        
        
    

    
        
      






class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        lengths = [item[2] for item in batch]
        sorted_index = torch.argsort(torch.tensor(lengths,dtype=torch.int), descending=True)
        lengths = np.array(lengths)[sorted_index]
        
        index = [item[3] for item in batch]
        sorted_index = torch.argsort(torch.tensor(index,dtype=torch.int), descending=True)
        index = np.array(index)[sorted_index]

        imgs = imgs[sorted_index]
        targets = targets[sorted_index]

        return imgs, targets, lengths, index


def get_loader(
    csv,
    imm_dir , 
    freq_threshold , 
    transform,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    drop_last=True,
):
    dataset  = CocoDataset(  csv, imm_dir, freq_threshold, transform)


    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,

        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset

