import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageDraw
from random import random, choice, randint
from torchvision import transforms

class Yolo_Dataset(Dataset):

    def __init__(self, S, B, C, IMG_SIZE, imgs_list):
        self.S = S
        self.B = B
        self.C = C
        self.IMG_SIZE = IMG_SIZE
        self.imgs_list = imgs_list
        self.annotations = pd.read_csv('annotations.csv')
        self.transformer = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self, index):
        
        img_path = self.imgs_list[index]
        annotations = self.annotations[self.annotations['img_path'] == img_path].values
        img_pil = Image.open(img_path)

        if (random() > 0.5):
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            annotations[:,4] = 1 - annotations[:,4]
        
        if (random() > 0.5):
            img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
            annotations[:,5] = 1 - annotations[:,5]
        
        img_tensor = self.transformer(img_pil)
        target_tensor = self.preparar_anotacoes(annotations)

        return img_tensor, target_tensor
    
    def preparar_anotacoes(self, annotations):

        target_tensor = torch.zeros((self.S, self.S, self.C + 5))
        cell_size = 1 / self.S

        for img_path, imgw, imgh, classe_ind, xc, yc, w, h in annotations:
            coluna, linha = int(xc / cell_size), int(yc / cell_size)
            xc_rel, yc_rel = (xc - coluna*cell_size)/cell_size, (yc - linha*cell_size)/cell_size
            w_rel, h_rel = w/cell_size, h/cell_size
            
            classes = torch.zeros(self.C)
            classes[classe_ind] = 1
            target_tensor[linha, coluna] = torch.hstack(
                [classes, torch.tensor([1, xc_rel, yc_rel, w_rel, h_rel])]
            )
        
        return target_tensor

def testar_anotacoes():
    df = pd.read_csv('annotations.csv')
    imgs_list = df['img_path'].unique()
    img_path = choice(imgs_list)
    img_pil = Image.open(img_path)
    draw = ImageDraw.Draw(img_pil)
    imgw, imgh = img_pil.size
    annotations = df[df['img_path'] == img_path].values
    for img_path,imgw,imgh,classe,xc,yc,w,h in annotations:
        xc, yc, w, h = xc*imgw, yc*imgh, w*imgw, h*imgh
        x1, y1, x2, y2 = xc-w/2, yc-h/2, xc+w/2, yc+h/2
        draw.rectangle([x1, y1, x2, y2], fill=None, outline='red', width=2)
    
    img_pil.show()

def testar_yolo_dataset():
    
    df = pd.read_csv('annotations.csv')
    imgs_list = df['img_path'].unique()

    S = 7
    B = 3
    C = 3
    IMG_SIZE = 400

    dataset = Yolo_Dataset(S, B, C, IMG_SIZE, imgs_list)
    img_tensor, target_tensor = choice(dataset)

    print (f'img_tensor.shape: {img_tensor.shape}, target_tensor.shape: {target_tensor.shape}')

    inv_transformer = transforms.Compose([
        transforms.Normalize(mean=(-1., -1., -1.), std=(2., 2., 2.)),
        transforms.ToPILImage()
    ])
    img_pil = inv_transformer(img_tensor)
    imgw, imgh = img_pil.size
    draw = ImageDraw.Draw(img_pil)

    cell_size = 1/S
    for linha in range(S):
        for coluna in range(S):
            vetor = target_tensor[linha, coluna]
            classes = vetor[:C]
            prob_classe, ind_classe = classes.max(0)
            prob_obj, xc, yc, w, h = vetor[C:]
            if (prob_classe > 0.5):
                xc, yc, w, h = xc*cell_size, yc*cell_size, w*cell_size, h*cell_size
                xc, yc = xc+cell_size*coluna, yc+cell_size*linha
                xc, yc, w, h = xc*imgw, yc*imgh, w*imgw, h*imgh
                x1, y1, x2, y2 = xc-w/2, yc-h/2, xc+w/2, yc+h/2
                draw.rectangle([x1, y1, x2, y2], fill=None, outline='red', width=2)
                draw.rectangle([x1, y1-15,x1+30, y1], fill='red')
                draw.text([x1+3, y1-13], str(ind_classe.item()))
    
    img_pil.show()

if (__name__ == '__main__'):
    testar_yolo_dataset()
    # testar_anotacoes() # -ok