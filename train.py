import torch
from model import Model
from loss import Yolo_Loss
from utils import preparar_dataloaders, validar_epoca, salvar_checkpoint, analisar_resultado_epoca
from argparse import ArgumentParser
from tqdm import tqdm
from colorama import Fore
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR

parser = ArgumentParser()
parser.add_argument('--s', type=int, default=4, help='Número de grids nas imagens.')
parser.add_argument('--imgsize', type=int, default=350, help='Redimensionamento das imagens.')
parser.add_argument('--e', type=int, default=10, help='Número de épocas.')
parser.add_argument('--lr', type=float, default=1e-4, help='LEARNING RATE.')
parser.add_argument('--batchsize', type=int, default=16, help='Tamanho do lote.')
parser.add_argument('--ct', type=float, default=0.5, help='Class threshold.')

args = parser.parse_args()
print (f'{Fore.RED}{args}{Fore.RESET}')

S = args.s
C = 3 # número de classes
IMG_SIZE = args.imgsize
EPOCHS = args.e
LEARNING_RATE = args.lr
BATCH_SIZE = args.batchsize
CLASS_THRESHOLD = args.ct

print ('preparando device...')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print (f'device: {DEVICE}')

print ('preparando modelo...')
model = Model(S, C, IMG_SIZE)
B = len(model.anchors)
model.to(DEVICE)
model.anchors = model.anchors.to(DEVICE)

print ('preparando otimizador')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = MultiStepLR(optimizer, [3, 10, 20, 30], gamma=0.1) # 1e-3, 1e-4, 1e-5, 1e-6

print ('preparando funcao perda...')
loss_fn = Yolo_Loss(S, B, C)
loss_fn.to(DEVICE)

print ('preparando dataloaders...')
train_dataloader, test_dataloader, train_dataset, test_dataset = preparar_dataloaders(S, B, C, IMG_SIZE, BATCH_SIZE, 0.1)

print ('iniciando treinamento')
for epoch in range(EPOCHS):

    model.train()
    train_loss, elementos = 0, 0

    for imgs_tensor, targets_tensor in tqdm(train_dataloader):
        imgs_tensor, targets_tensor = imgs_tensor.to(DEVICE), targets_tensor.to(DEVICE)
        
        optimizer.zero_grad()
        predicts_tensor = model(imgs_tensor)
        loss = loss_fn(predicts_tensor, targets_tensor, model.anchors)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        elementos += len(imgs_tensor)
    
    scheduler.step()
    
    train_loss = train_loss / elementos
    test_loss = validar_epoca(test_dataloader, loss_fn, model, DEVICE)
    print (Fore.YELLOW + f'epoca: {epoch}, train_loss: {round(train_loss, 3)}, test_loss: {round(test_loss, 3)}' + Fore.RESET)
    salvar_checkpoint(model, epoch)
    analisar_resultado_epoca(test_dataloader, model, epoch, class_threshold=CLASS_THRESHOLD, device=DEVICE)