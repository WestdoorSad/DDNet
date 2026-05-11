import os
import time
import pprint
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)


def strip_thop_from_state_dict(state_dict):
    """去掉 thop.profile 写入的 total_ops / total_params，避免 load_state_dict 报错。"""
    if not state_dict:
        return state_dict
    return {
        k: v for k, v in state_dict.items()
        if not (k.endswith('total_ops') or k.endswith('total_params'))
    }


def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)

def calculate_accuracy_each_snr(Y, Y_hat, Z, mods=None, epoch=None):
    Z_array = Z
    snrs = sorted(list(set(Z_array)))
    # snrs = np.arange(-20,32,2)
    acc = np.zeros(len(snrs))
    Y_index = Y
    Y_index_hat = Y_hat
    i = 0
    for snr in snrs:
        Y_snr = Y_index[np.where(Z_array == snr)]
        Y_hat_snr = Y_index_hat[np.where(Z_array == snr)]
        acc[i] = np.sum(Y_snr == Y_hat_snr) / Y_snr.shape[0]
        i = i + 1

    print(snrs)
    for snr_acc in acc:
        print(snr_acc, end=', ')
    print()

    a = []
    b = []
    for snr in snrs[:]:
        Y_snr = Y_index[np.where(Z_array == snr)]
        Y_hat_snr = Y_index_hat[np.where(Z_array == snr)]
        a += Y_snr.tolist()
        b += Y_hat_snr.tolist()
    print(snrs[:len(snrs) // 3], 'acc', np.sum(np.array(a) == np.array(b)) / np.array(b).shape[0] * 100)

    a = []
    b = []
    for snr in snrs[-len(snrs) // 3:]:
        Y_snr = Y_index[np.where(Z_array == snr)]
        Y_hat_snr = Y_index_hat[np.where(Z_array == snr)]
        a += Y_snr.tolist()
        b += Y_hat_snr.tolist()
    print(snrs[-len(snrs) // 3:], 'acc', np.sum(np.array(a) == np.array(b)) / np.array(b).shape[0] * 100)

    a = []
    b = []
    for snr in snrs[len(snrs) // 3:-len(snrs) // 3]:
        Y_snr = Y_index[np.where(Z_array == snr)]
        Y_hat_snr = Y_index_hat[np.where(Z_array == snr)]
        a += Y_snr.tolist()
        b += Y_hat_snr.tolist()
    print(snrs[len(snrs) // 3:-len(snrs) // 3], 'acc', np.sum(np.array(a) == np.array(b)) / np.array(b).shape[0] * 100)

    # Draw_Heatmap(Y_index[np.where(Z_array == 2)], Y_index_hat[np.where(Z_array == 2)], mods, 'experiments/{}_snr_2dB.png'.format(epoch))
    # Draw_Heatmap(Y_index[np.where(Z_array == 6)], Y_index_hat[np.where(Z_array == 6)], mods, 'experiments/{}_snr_6dB.png'.format(epoch))

    plt.figure(figsize=(8, 6))
    plt.plot(snrs, acc, label='test_acc')
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10a")
    plt.legend()
    plt.grid()
    # plt.show()

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
def Draw_Heatmap(true_labels, pred_labels, mods, save_path, fs=5):

    sk_cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels)

    sk_cm = sk_cm.astype('float') / sk_cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(fs, fs))
    ax = plt.subplot(111)
    # ax.set_title("Confusion Matrix", size=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16  

    heatmap = sns.heatmap(sk_cm, linewidths=1, cmap='Blues', linecolor='white', cbar=False,
                annot=True, xticklabels=mods, yticklabels=mods, ax=ax, fmt=".2f")


    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=90)

    for label in heatmap.get_xticklabels():
        label.set_fontsize(17)  
    for label in heatmap.get_yticklabels():
        label.set_fontsize(17)  

    # plt.xlabel('Predicted Label', size=10)
    # plt.ylabel('True Label', size=10)
    plt.subplots_adjust(right=1, top=1)
    fig.savefig(save_path)

    return save_path
