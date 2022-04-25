from importlib.resources import path
from PIL import Image, ImageDraw
from random import random, choice, randint
import pathlib
import os
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser

def random_color():
    return (randint(0, 150), randint(0, 150), randint(0, 150))

def create_image(n_objects):
    # backgrounds_list = [str(p) for p in pathlib.Path('./backgrounds').glob('*.jpg')]

    imgw, imgh = randint(300, 400), randint(300, 400)
    img_pil = Image.new('RGB', size=(imgw, imgh), color='white')

    draw = ImageDraw.Draw(img_pil)
    annotations = []

    for _ in range(90):
        x1, y1, x2, y2 = randint(0, imgw), randint(0, imgh), randint(0, imgw), randint(0, imgh)
        draw.line([x1, y1, x2, y2], fill=random_color(), width=1)

    for n in range(n_objects):
        
        w, h = randint(30, 50), randint(30, 50)
        x1, y1 = randint(0, imgw-w), randint(0, imgh-h)
        x2, y2 = x1+w, y1+h
        xc, yc = x1+w/2, y1+h/2

        tipo = choice([0, 1, 2])
        if (tipo == 0): draw.rectangle([x1, y1, x2, y2], fill=random_color())
        if (tipo == 1): draw.ellipse([x1, y1, x2, y2], fill=random_color())
        if (tipo == 2):
            p0, p1, p2 = (x1, y2), (x2, y2), (xc,y1)
            draw.polygon([p0, p1, p2], fill=random_color())
        
        annotations.append([imgw, imgh, tipo, xc/imgw, yc/imgh, w/imgw, h/imgh])
    
    return img_pil, annotations

def create_images_dataset(n_images):

    imgs_dir = './imgs/'
    if (not(os.path.exists(imgs_dir))): os.mkdir(imgs_dir)

    df_annotations = []
    for n in tqdm(range(n_images)):
        img_path = f'{imgs_dir}img_{n}.jpg'
        img_pil, annotations = create_image(randint(1, 5))
        for imgw, imgh, classe, xc, yc, w, h in annotations:
            df_annotations.append([img_path, imgw, imgh, classe, xc, yc, w, h])
        img_pil.save(img_path)
    
    df_annotations = pd.DataFrame(df_annotations, columns=['img_path', 'imgw', 'imgh', 'classe', 'xc', 'yc', 'w', 'h'])
    df_annotations.to_csv('annotations.csv', index=False)
    print (f'Dataset criado!')

if (__name__ == '__main__'):
    parser = ArgumentParser()
    parser.add_argument('--n', type=int, help='Número de imagens no dataset.')
    args = parser.parse_args()
    n = args.n

    create_images_dataset(n_images=n)