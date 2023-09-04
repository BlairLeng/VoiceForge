from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib


def visualization_np(inputs: np.ndarray):
    cols, raws = inputs.shape
    if raws == 192:
        inputs = inputs.transpose()

    # 进行 tSNE 计算
    tsne = TSNE(n_components=2, random_state=0)
    reduced_data = tsne.fit_transform(inputs)
    # print(reduced_data.shape)

    # 方便可视化不同的颜色
    labels = np.random.randint(0, 109, 256)

    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='jet', s=50)
    plt.colorbar(scatter)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('2D t-SNE visualization of the data')
    plt.show()

