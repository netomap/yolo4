import torch
from torch import nn

class Model(nn.Module):

    def __init__(self, S, B, C, IMG_SIZE, architecture_config = None):
        super(Model, self).__init__()

        self.S = S
        self.B = B
        self.C = C
        self.IMG_SIZE = IMG_SIZE
        architecture_default = [
            'conv2d,3,32,3,2,0',
            'leakyrelu,0.2',
            'conv2d,32,32,3,2,0',
            'leakyrelu,0.2',
            'conv2d,32,32,3,2,0',
            'leakyrelu,0.2',
            'conv2d,32,32,3,2,0',
            'leakyrelu,0.2',
            'conv2d,32,32,3,2,0',
            'leakyrelu,0.2',
            'conv2d,32,32,3,2,0',
            'leakyrelu,0.2',
            'flatten,',
            'linear,-1,1024',
            'leakyrelu,0.2',
            'linear,1024,-1'
        ]
        self.architecture_config = architecture_config if (architecture_config) else architecture_default

        self.net = nn.Sequential(*self.create_blocks())
    
    def calcular_n_features(self, blocks):
        model_test = nn.Sequential(*blocks)
        input_test = torch.rand((1, 3, self.IMG_SIZE, self.IMG_SIZE))
        output = model_test(input_test)
        return output.shape[-1]

    def create_blocks(self):
        blocks = []
        in_features = -1

        for block in self.architecture_config:
            b = block.split(',')
            tipo = b[0]

            if (tipo == 'conv2d'):
                blocks.append(nn.Conv2d(int(b[1]), int(b[2]), int(b[3]), int(b[4]), int(b[5])))
            
            if (tipo == 'leakyrelu'):
                blocks.append(nn.LeakyReLU(float(b[1])))
            
            if (tipo == 'flatten'):
                blocks.append(nn.Flatten())
                in_features = self.calcular_n_features(blocks)
            
            if (tipo == 'linear'):
                in_f, out_f = int(b[1]), int(b[2])
                in_f = in_f if (in_f != -1) else in_features
                out_f = out_f if (out_f != -1) else self.S*self.S*self.B*(self.C+5)
                blocks.append(nn.Linear(in_f, out_f))
        
        return blocks
    
    def forward(self, x):
        return self.net(x)

def testar_modelo():
    model = Model(4, 3, 3, 400)
    print (model)
    input_test = torch.rand((1, 3, 400, 400))
    output = model(input_test)
    print (f'output.shape: {output.shape}')

if __name__ == '__main__':
    testar_modelo()