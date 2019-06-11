#coding:utf8
import sys
sys.path.append("../")
import os
from pfa_build.pfa import *
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def visualize_pfa():
    pm_file_path = "/Users/dong/Desktop/ijcai19/best_dtmc/GRU60000.pm"
    label_path = "/Users/dong/Desktop/ijcai19/best_dtmc/GRU60000_label.txt"
    pfa = PFA(pm_file_path=pm_file_path, label_path=label_path, integrate_files_path=None)
    pfa.generate_grpah(save_path="/Users/dong/Documents/bitbucket/pfa001/document/paper/exp_data/dtmc.pdf", view=True, layout="circo")

def visualize_clusters():
    fp="/Users/dong/Desktop/ijcai实验结果/best_dtmc/points_tsne.pkl"
    with open(fp,'r') as f:
        X_embedded=pickle.load(f)
    with PdfPages("/Users/dong/Desktop/grapg_cluster.pdf") as pdf:
        plt.scatter(X_embedded[:,0],X_embedded[:,1])
        pdf.savefig()
        plt.close()

def saveRformat(filePath,savePath):

    word_count = {}
    with open(filePath,"r") as f:
        for line in f.readlines():
            data = [ele.strip() for ele in line.split(",")]
            word, frq = data[0],int(data[2])
            word = word.lower()
            if word in word_count.keys():
                word_count[word]+=frq
            else:
                word_count[word]=frq

    with open(savePath,"w") as f:
        items=word_count.items()
        items.sort(key=lambda x:x[1],reverse=True)
        for item in items:
            f.write("{} {}\n".format(item[0],item[1]))

if __name__=="__main__":
    # visualize_clusters()
    visualize_pfa()
    # filePath = "/Users/dong/Desktop/ijcai19/case_study/key-words/semantic_result/pos_count.txt"
    # savePath = "/Users/dong/Desktop/ijcai19/case_study/key-words/semantic_result/R2show/pos_count.txt"
    # saveRformat(filePath,savePath)





