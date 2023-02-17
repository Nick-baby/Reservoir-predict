from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def plot_embedding_tsne(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        if label[i]==0:
            color = "red"
            cirl="s"
            plt.scatter(data[i, 0], data[i, 1], color=color, alpha=0.6 )
            # plt.legend(["red"])
        else:
            color="blue"
            cirl = "t"
            plt.scatter(data[i, 0], data[i, 1], color=color, alpha=0.6)
            # plt.legend(["Blue"])
        """plt.text(data[i, 0], data[i, 1], cirl,
                 color=color,
                 # color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})"""
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel('t-SNE 1d', fontsize=32)
    plt.ylabel('t-SNE 2d', fontsize=32)
    return fig
def plot_embedding_tsne_1d(src, tar):

    fig = plt.figure()
    # ax = plt.subplot(111)
    import seaborn as sns
    from scipy import stats


    sns.distplot(src, kde=True, bins=50,color="#ef4026",
                     kde_kws={"label": "Source", 'linestyle': "solid"})
    sns.distplot(tar, kde=True, bins=50,color="#0339f8",
                     kde_kws={"label": "A1", 'linestyle': "-."})
    plt.legend(loc="upper right", fontsize=13)
    plt.xlabel('Feature', fontsize=15, weight='bold')
    plt.ylabel("Density", fontsize=15, weight='bold')


def main():
    import pandas as pd
    data, label, n_samples, n_features = get_data()
    # all_data = pd.read_excel(source_location).values[:, 1:]
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         '特征数据分布可视化'
                         )
    plt.show()

def tsne(src,tar,epoch):
    # data, label, n_samples, n_features = get_data()
    # data=np.concatenate((np.array(src),np.array(tar)),axis=0)
    src=np.concatenate(src)
    tar=np.concatenate(tar)[:360]

    np.random.seed(6)
    np.random.shuffle(src)

    src=src[:1600]
    tar=tar[:360]

    data=np.concatenate((src,tar),axis=0)
    label=np.concatenate((np.zeros(src.shape[0]),np.ones(tar.shape[0])),axis=0)

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding_tsne(result, label,
                         'POR of well A1'
                         )

    plt.savefig(f"tsnePlot/fig_ep{epoch}"+'.svg',format='svg', bbox_inches='tight')
    plt.savefig(f"tsnePlot/fig_ep{epoch}")
def tsne_1d(src,tar,epoch):
    # data, label, n_samples, n_features = get_data()
    # data=np.concatenate((np.array(src),np.array(tar)),axis=0)
    src=np.concatenate(src)
    tar=np.concatenate(tar)

    np.random.seed(6)
    np.random.shuffle(src)

    src=src[0:360]
    tar=tar[0:360]

    data=np.concatenate((src,tar),axis=0)
    label=np.concatenate((np.zeros(src.shape[0]),np.ones(tar.shape[0])),axis=0)

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=1, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    # src = tsne.fit_transform(src)
    # tar = tsne.fit_transform(tar)
    src=result[0:360]
    tar=result[360:]


    plot_embedding_tsne_1d(src, tar)

    plt.savefig(f"tsnePlot/fig_ep{epoch}"+'.svg',format='svg', bbox_inches='tight')
    plt.savefig(f"tsnePlot/fig_ep{epoch}")


