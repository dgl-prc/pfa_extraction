import os
import torch
import pickle
'''
these functions are project-free.
'''
def save_readme(parent_path, content):
    with open(os.path.join(parent_path, "README"), "w") as f:
        f.writelines(content)

def save_model(model, train_acc, test_acc, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_acc = "{0:.4f}".format(train_acc)
    test_acc = "{0:.4f}".format(test_acc)
    save_file = os.path.join(save_path, 'train_acc-' + train_acc + '-test_acc-' + test_acc + '.pkl')
    torch.save(model.state_dict(), save_file)

def make_check_point_folder(task_name,dataset, modelType):
    check_point_folder = os.path.join("./tmp",task_name, dataset, modelType)
    if not os.path.exists(check_point_folder):
        os.makedirs(check_point_folder)
    return check_point_folder

def converPickle2python2(pickle_file):
    with open(pickle_file, "rb") as f:
        obj = pickle.load(f)

    file_name, file_exten = os.path.splitext(pickle_file)
    new_file_name = "".join([file_name, "protocol2", file_exten])
    with open(new_file_name, "wb") as f:
        pickle.dump(obj, f, protocol=2)

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        pkl_obj = pickle.load(f)
    return pkl_obj

def save_pickle(file_path, obj, protocol=3):
    parent_path = os.path.split(file_path)[0]
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)