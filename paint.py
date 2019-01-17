import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_function(outputs, label, type_list):    
    outputs['flag'] = list(map(lambda x: True if x in cha else False, outputs['label']))
    outputs = outputs[outputs['flag'] == True]
    outputs = outputs.reset_index(drop = True)
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(outputs.iloc[:, 0:339])
    plt.figure(figsize=(12, 12))
    plt.grid(True)
    plt.scatter(t[0], t[1], c = outputs['label']*1, marker = None, alpha = 0.3)
    
def paint_draw(picture, num):
    pic = Image.new("RGB",(x,y))
    for row in range(x):
        for column in range(y):
            if max(picture[0][i][j], picture[1][i][j], picture[2][i][j]) <= 0:
                pic.putpixel([i,j], (255,255,255))
            else:
                pic.putpixel([i,j], (0,0,0))
     pic.save("./picture" + str(num)+ ".png")


if __name__ == '__main__':
    labels = pd.DataFrame(np.load('labels.npy'), columns = ['label'])
    outputs = pd.DataFrame(np.vstack((np.load('outputs_01.npy'), np.load('outputs_02.npy'))))
    outputs['label'] = labels['label']
    type_list = [3, 10, 15, 31, 33, 64, 66, 93, 83, 118]
    tsne_function(outputs, label, type_list)
    pic = np.load('pic.npy')
    num = 0
    for picture in pic:
        paint_draw(picture)  
        num +=1
    