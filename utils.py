import torch
from dataset import Yolo_Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime
from torchvision import transforms
from random import choice
from PIL import ImageDraw

def carregar_modelo(checkpoint_path):
    from model import Model
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    S = checkpoint['S']
    anchors = checkpoint['anchors']
    C = checkpoint['C']
    IMG_SIZE = checkpoint['IMG_SIZE']
    architecture_config = checkpoint['architecture_config']

    model = Model(S, C, IMG_SIZE, architecture_config=architecture_config)
    model.anchors = anchors
    model.B = len(model.anchors)
    print (model.load_state_dict(checkpoint['state_dict']))

    return model

def preparar_dataloaders(S, B, C, IMG_SIZE, BATCH_SIZE, test_size):
    df = pd.read_csv('annotations.csv')
    imgs = df['img_path'].unique()
    imgs_train, imgs_test = train_test_split(imgs, test_size=test_size, shuffle=True)
    train_dataset = Yolo_Dataset(S, B, C, IMG_SIZE, imgs_train)
    test_dataset = Yolo_Dataset(S, B, C, IMG_SIZE, imgs_test)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print (f'train_dataset: {len(train_dataset)} images.')
    print (f'test_dataset: {len(test_dataset)} images.')
    print (f'train_dataloader: {len(train_dataloader)} bathces.')
    print (f'test_dataloader: {len(test_dataloader)} batches.')
    return train_dataloader, test_dataloader, train_dataset, test_dataset

def validar_epoca(test_dataloader, loss_fn, model, DEVICE):
    print ('validando epoca...')
    model.eval()
    with torch.no_grad():
        test_loss, elementos = 0, 0
        for imgs_tensor, targets_tensor in tqdm(test_dataloader):
            imgs_tensor, targets_tensor = imgs_tensor.to(DEVICE), targets_tensor.to(DEVICE)
            predicts_tensor = model(imgs_tensor)
            loss = loss_fn(predicts_tensor, targets_tensor, model.anchors)
            test_loss += loss.item()
            elementos += len(imgs_tensor)

    return test_loss / elementos

def salvar_checkpoint(model, epoch):
    
    checkpoints_dir = './checkpoints/'
    if (not(os.path.exists(checkpoints_dir))): os.mkdir(checkpoints_dir)

    checkpoint = {
        'S': model.S,
        'anchors': model.anchors,
        'C': model.C,
        'IMG_SIZE': model.IMG_SIZE,
        'state_dict': model.state_dict(),
        'architecture_config': model.architecture_config,
        'epoch': epoch,
        'datetime': datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M')
    }
    torch.save(checkpoint, f'{checkpoints_dir}checkpoint_{epoch}.pth')

def analisar_resultado_epoca(dataloader_, model, epoch, class_threshold=0.5, device = torch.device('cpu')):

    inv_transformer = transforms.Compose([
        transforms.Normalize(mean=(-1., -1., -1.), std=(2., 2., 2.)),
        transforms.ToPILImage()
    ])

    S = model.S
    B = model.B
    cell_size = 1/S
    C = model.C

    model.eval()
    with torch.no_grad():
        imgs_tensor, targets_tensor = next(iter(dataloader_))
        predicts_tensor = model(imgs_tensor.to(device))

        predicts_tensor = predicts_tensor.reshape((-1, S, S, B, C+5))
        predicts_tensor[..., C+1:C+3] = torch.sigmoid(predicts_tensor[..., C+1:C+3])
        predicts_tensor[..., -2:] = torch.exp(predicts_tensor[..., -2:]) * model.anchors
        img_tensor, predict_tensor = choice(list(zip(imgs_tensor, predicts_tensor)))
    
        img_pil = inv_transformer(img_tensor)
        imgw, imgh = img_pil.size
        draw = ImageDraw.Draw(img_pil)

        anotacoes = []

        for linha in range(S):
            for coluna in range(S):
                bboxes = predict_tensor[linha, coluna]
                bboxes[:, -4:] = bboxes[:, -4:] * cell_size # multiplica xc, yc, w, h por cell_size
                bboxes[:, -4] = bboxes[:, -4] + coluna*cell_size # soma xc com offset X
                bboxes[:, -3] = bboxes[:, -3] + linha*cell_size # soma yc com offset Y
                for box in bboxes:
                    classes = box[:C]
                    classes = torch.softmax(classes, dim=-1)
                    prob_class, ind_class = classes.max(0)
                    if (prob_class > class_threshold):
                        prob_obj = box[C]
                        xc, yc, w, h = box[-4:]
                        x1, y1, x2, y2 = xc-w/2, yc-h/2, xc+w/2, yc+h/2
                        x1, y1, x2, y2 = x1*imgw, y1*imgh, x2*imgw, y2*imgh
                        anotacoes.append([prob_class, ind_class.item(), x1, y1, x2, y2])
                        draw.rectangle([x1, y1, x2, y2], fill=None, outline='red', width=1)
                        draw.rectangle([x1, y1-15, x1+50, y1], fill='red')
                        draw.text([x1+2, y1-13], str(ind_class.item()) + '-' + str(int(100*prob_class.item())) + "%", fill='white')
    
    results_dir = './img_results/'
    if (not(os.path.exists(results_dir))): os.mkdir(results_dir)

    img_pil.save(f'{results_dir}result_{epoch}.jpg')

def predict(model, img_pil, class_threshold=0.5):

    transformer = transforms.Compose([
        transforms.Resize((model.IMG_SIZE, model.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    S = model.S
    B = model.B
    cell_size = 1/S
    C = model.C

    model.eval()

    imgs_tensor = transformer(img_pil).unsqueeze(0)
    predicts_tensor = model(imgs_tensor)

    predicts_tensor = predicts_tensor.reshape((-1, S, S, B, C+5))
    predicts_tensor[..., C+1:C+3] = torch.sigmoid(predicts_tensor[..., C+1:C+3])
    predicts_tensor[..., -2:] = torch.exp(predicts_tensor[..., -2:]) * model.anchors
    
    predict_tensor = predicts_tensor[0]

    imgw, imgh = img_pil.size
    draw = ImageDraw.Draw(img_pil)

    anotacoes = []

    for linha in range(S):
        for coluna in range(S):
            bboxes = predict_tensor[linha, coluna]
            bboxes[:, -4:] = bboxes[:, -4:] * cell_size # multiplica xc, yc, w, h por cell_size
            bboxes[:, -4] = bboxes[:, -4] + coluna*cell_size # soma xc com offset X
            bboxes[:, -3] = bboxes[:, -3] + linha*cell_size # soma yc com offset Y
            for box in bboxes:
                classes = box[:C]
                classes = torch.softmax(classes, dim=-1)
                prob_class, ind_class = classes.max(0)
                if (prob_class > class_threshold):
                    prob_obj = box[C]
                    xc, yc, w, h = box[-4:]
                    x1, y1, x2, y2 = xc-w/2, yc-h/2, xc+w/2, yc+h/2
                    x1, y1, x2, y2 = x1*imgw, y1*imgh, x2*imgw, y2*imgh
                    anotacoes.append([prob_class, ind_class.item(), x1, y1, x2, y2])
                    draw.rectangle([x1, y1, x2, y2], fill=None, outline='red', width=1)
                    draw.rectangle([x1, y1-15, x1+50, y1], fill='red')
                    draw.text([x1+2, y1-13], str(ind_class.item()) + '-' + str(int(100*prob_class.item())) + "%", fill='white')
    
    return img_pil, anotacoes