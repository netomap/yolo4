from dataset import Yolo_Dataset
from model import Model
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.ops.boxes import _box_cxcywh_to_xyxy, box_iou
from torch.functional import F

def criar_offsets(N_BATCH, S, B):
    cell_size = 1/S
    vetor = []
    for linha in range(S):
        for coluna in range(S):
            if (B != -1): # menos 1 representa que serve para criar offset para o tensor targets
                vetor.append(torch.tensor([coluna*cell_size, linha*cell_size]).repeat((B,1)))
            else:
                vetor.append(torch.tensor([coluna*cell_size, linha*cell_size]))
    
    if (B != -1):
        return torch.vstack(vetor).repeat((N_BATCH,1)).reshape((N_BATCH, S, S, B, 2))
    else:
        return torch.vstack(vetor).repeat((N_BATCH,1)).reshape((N_BATCH, S, S, 2))

class Yolo_Loss(nn.Module):
    
    def __init__(self, S, B, C):
        self.S = S
        self.B = B
        self.C = C
        super(Yolo_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
    
    def forward(self, predicts_tensor, targets_tensor, anchors):
        
        predicts_tensor = predicts_tensor.reshape((-1, self.S, self.S, self.B, self.C+5))

        # ESCOLHENDO A MELHOR PREDIÇÃO DE CADA CÉLULA ===================================

        predicts_tensor[..., self.C+1:self.C+3] = torch.sigmoid(predicts_tensor[..., self.C+1:self.C+3])
        # as posições de xc, yc recebem sigmoid e depois somam com seus offsets

        predicts_tensor[..., -2:] = torch.exp(predicts_tensor[...,-2:]) * anchors
        # as posições de w, h recebem exponencial junto com os produtos de anchors

        predicts_bboxes = _box_cxcywh_to_xyxy(predicts_tensor[..., -4:].reshape((-1, 4)))
        # prepara vetor e transforma xcycwh => x1y1x2y2
        predicts_bboxes = predicts_bboxes.reshape((-1, self.B, 4)) # refatora para cada elemento ter self.B boxes para comparar melhor iou
        
        targets_bboxes = _box_cxcywh_to_xyxy(targets_tensor[..., -4:]).reshape((-1, 1, 4))
        # prepara vetor e transforma xcycwh => x1y1x2y2

        ious = torch.stack([box_iou(p, t).reshape(-1) for p, t in list(zip(predicts_bboxes, targets_bboxes))])
        # calcula ious entre cada bbox do predict com a única opção de target para cada célula

        best_ious, best_indices = ious.max(dim=-1) # melhores índices e valores
        best_ious = best_ious.reshape((-1, self.S, self.S))
        best_indices = best_indices.reshape((-1, self.S, self.S))

        one_hot_tensor = F.one_hot(best_indices, num_classes=self.B).unsqueeze_(-1)
        predicts_tensor = predicts_tensor.masked_select(one_hot_tensor == 1).reshape((-1, self.S, self.S, self.C+5))
        predicts_tensor[..., 0] = predicts_tensor[..., 0] * best_indices
        # ESCOLHENDO A MELHOR PREDIÇÃO DE CADA CÉLULA ===================================

        exists_box = targets_tensor[..., self.C].unsqueeze(-1) # fica na posição C

        # PERDA PARA XC, YC =============================================================
        predicts_xcyc = predicts_tensor[..., self.C+1:self.C+3]
        targets_xcyc = targets_tensor[..., self.C+1:self.C+3]

        loss_xcyc = self.mse(
            exists_box * predicts_xcyc,
            exists_box * targets_xcyc
        )
        # PERDA PARA XC, YC =============================================================

        # PERDA PARA W, H ++=============================================================
        predicts_wh = predicts_tensor[..., -2:]
        predicts_wh = torch.sign(predicts_wh) * torch.sqrt(torch.abs(predicts_wh) + 1e-6)
        targets_wh = targets_tensor[..., -2:]
        loss_wh = self.mse(
            exists_box * predicts_wh,
            exists_box * targets_wh
        )
        # PERDA PARA W, H ++=============================================================

        # PERDA PARA CLASSES ============================================================
        predicts_classes = predicts_tensor[..., :self.C]
        targets_classes = targets_tensor[..., :self.C]
        loss_classes = self.mse(
            exists_box * predicts_classes,
            exists_box * targets_classes
        )
        # PERDA PARA CLASSES ============================================================

        predicts_prob_obj = predicts_tensor[...,self.C:self.C+1]
        targets_prob_obj = targets_tensor[..., self.C:self.C+1]
        
        # PERDA PARA PROB_OBJ ===========================================================
        loss_obj = self.mse(
            exists_box * predicts_prob_obj,
            exists_box * targets_prob_obj
        )
        # PERDA PARA PROB_OBJ ===========================================================

        # PERDA PARA NO_OBJ CLASSES =====================================================
        no_obj_box = 1 - exists_box
        loss_noobj_classes = self.mse(
            no_obj_box * predicts_prob_obj,
            no_obj_box * targets_prob_obj
        )
        # PERDA PARA NO_OBJ CLASSES =====================================================

        # print (f'({loss_xcyc=} \n {loss_wh=} \n {loss_classes=} \n{loss_noobj_classes=} \n {loss_obj=})')
        
        return (loss_xcyc + loss_wh + loss_classes + loss_noobj_classes + loss_obj)

def testar_loss():

    S = 4
    C = 3
    IMG_SIZE = 350
    model = Model(S, C, IMG_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    B = model.B

    df = pd.read_csv('annotations.csv')
    img_list = df['img_path'].unique()
    dataset = Yolo_Dataset(S, B, C, IMG_SIZE, img_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss_fn = Yolo_Loss(S, B, C)

    for _ in range(5):
        optimizer.zero_grad()
        imgs_tensor, targets_tensor = next(iter(dataloader))
        predicts_tensor = model(imgs_tensor)
        loss = loss_fn(predicts_tensor, targets_tensor, model.anchors)
        loss.backward()
        print (loss.item())
        optimizer.step()
    
if __name__ == '__main__':
    testar_loss()