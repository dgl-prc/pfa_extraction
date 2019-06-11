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


def draw_pfa(save_path):

    starts = ["start","start","s1","s1","s2","s2"]
    ends =   ["s1",    "s2", "s1","s2","s1","s2"  ]
    actions = ["a",     "b",  "b","a", "b", "a"]
    wight = [0.8,0.2,0.4,0.6,0.3,0.7]
    mygraph = gv.Digraph('DTMC', engine="dot",graph_attr={"rankdir":"LR"})
    for start, end, weight,label in zip(starts,ends,wight,actions):
        mygraph.node(start,shape="circle", margin="0.05")
        mygraph.node(end,shape="circle", margin="0.05")
        mygraph.edge(start, end, str(round(weight, 5))+"/"+label)
        mygraph.render(save_path, view=True)


class GraphNode(object):
        def __init__(self,name,label):
            self.name = name
            self.label = label



def draw_HCA(save_path):

    a=GraphNode("a","a")
    b=GraphNode("b","b")
    c=GraphNode("c","c")
    d = GraphNode("d", "d")
    ab = GraphNode("ab", "ab")
    c1 = GraphNode("c1", "c")
    d1 = GraphNode("d1", "d")
    abc = GraphNode("abc", "abc")
    d2=GraphNode("d2","d")
    abcd = GraphNode("abcd", "abcd")

    starts=  [a, b,  c, d,  ab, c1, d1, abc, d2]
    ends =   [ab,ab, c1,d1, abc,abc,d2, abcd,abcd ]

    # mygraph = gv.Digraph('tree', engine="dot",graph_attr={"rotate":90})
    mygraph = gv.Digraph('tree', engine="dot")
    for start, end in zip(starts,ends):
        mygraph.node(start.name,label=start.label,margin="0.001")
        mygraph.node(end.name,label=end.label,margin="0.001")
        mygraph.edge(start.name, end.name)
        mygraph.render(save_path, view=True)




def visualize_pfa(pm_file_path,label_path,save_path,layout="dot"):
    pfa = PFA(pm_file_path=pm_file_path, label_path=label_path, integrate_files_path=None)
    pfa.generate_grpah(save_path=save_path, view=True, layout=layout)


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
    pm_file_path="../casestudy/pfa/GRU60000.pm"
    label_path="../casestudy/pfa/GRU60000_label.txt"
    # save_path="../casestudy/pfa-refine"
    # draw_pfa(save_path)
    # save_path="../casestudy/hca"
    # draw_HCA(save_path)
    save_path="../casestudy/dtmc"
    visualize_pfa(pm_file_path,label_path,save_path)
    # filePath = "/Users/dong/Desktop/ijcai19/case_study/key-words/semantic_result/pos_count.txt"
    # savePath = "/Users/dong/Desktop/ijcai19/case_study/key-words/semantic_result/R2show/pos_count.txt"
    # saveRformat(filePath,savePath)





