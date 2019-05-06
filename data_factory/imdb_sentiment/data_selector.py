import os
from shutil import copyfile
import numpy as np
from imdb_data_process import *


def copy_files(files_list, src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file_name in files_list:
        copyfile(os.path.join(src_folder, file_name), os.path.join(dest_folder, file_name))


def copy_files_with_filter(files_list, src_folder, dest_folder, dataprocessor,max_size=250,max_length=100):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    cnt = 0
    for file_name in files_list:
        source_file = os.path.join(src_folder, file_name)
        dest_file = os.path.join(dest_folder, file_name)
        with open(source_file, "r") as f:
            sequence = f.readlines()
        word_sequence = map(dataprocessor.sequence_purifier, sequence)[0]
        if len(word_sequence) <= 100:  # only reserve the number of valid words is less than or equal to 100
            copyfile(source_file, dest_file)
            cnt+=1
        if cnt > max_size:
            break
    print("Success:{}".format(cnt))


def randomly_choose_train_test_data(random_seed, train_size, test_size, source_folder, dest_folder):
    train_neg_path = os.path.join(source_folder, 'train', 'neg')
    train_pos_path = os.path.join(source_folder, 'train', 'pos')
    test_neg_path = os.path.join(source_folder, 'test', 'neg')
    test_pos_path = os.path.join(source_folder, 'test', 'pos')

    train_neg_files_list = os.listdir(train_neg_path)
    train_pos_files_iist = os.listdir(train_pos_path)
    test_neg_files_list = os.listdir(test_neg_path)
    test_pos_files_iist = os.listdir(test_pos_path)

    rnd_train_pos_path = os.path.join(dest_folder, 'train', 'pos')
    rnd_train_neg_path = os.path.join(dest_folder, 'train', 'neg')
    rnd_test_pos_path = os.path.join(dest_folder, 'test', 'pos')
    rnd_test_neg_path = os.path.join(dest_folder, 'test', 'neg')

    np.random.seed(random_seed)
    random_train_neg = np.random.choice(train_neg_files_list, int(train_size * 0.5), replace=False)
    np.random.seed(random_seed)
    random_train_pos = np.random.choice(train_pos_files_iist, int(train_size * 0.5), replace=False)
    np.random.seed(random_seed)
    random_test_neg = np.random.choice(test_neg_files_list, int(test_size * 0.5), replace=False)
    np.random.seed(random_seed)
    random_test_pos = np.random.choice(test_pos_files_iist, int(test_size * 0.5), replace=False)

    # do copy
    copy_files(random_train_neg, train_neg_path, rnd_train_neg_path)
    copy_files(random_train_pos, train_pos_path, rnd_train_pos_path)
    copy_files(random_test_neg, test_neg_path, rnd_test_neg_path)
    copy_files(random_test_pos, test_pos_path, rnd_test_pos_path)

    print('DONE!')


def generate_data_large(train_size, test_size, random_seed, data_group, source_data, save_root):
    dest_folder = os.path.join(save_root, str(train_size) + '/pfa_expe' + str(data_group))
    randomly_choose_train_test_data(random_seed, train_size, test_size, source_data, dest_folder)


def make_data_for_PFA():
    train_size = 5000
    test_size = 1000
    data_root = "/home/dgl/project/pfa-data-generator/data/aclImdb"
    save_root = "/home/dgl/project/pfa-data-generator/data/exp_ijcai19"
    randomSeed1 = 5566
    randomSeed2 = 5577
    randomSeed3 = 5588
    randomSeed4 = 5599
    randomSeed5 = 55100
    generate_data_large(train_size, test_size, randomSeed1, 1, data_root, save_root)
    generate_data_large(train_size, test_size, randomSeed2, 2, data_root, save_root)
    generate_data_large(train_size, test_size, randomSeed3, 3, data_root, save_root)
    generate_data_large(train_size, test_size, randomSeed4, 4, data_root, save_root)
    generate_data_large(train_size, test_size, randomSeed5, 5, data_root, save_root)

    print('DONE!')


def make_data_for_DFA():
    '''
    For the DFA proposed in ICML 2018.
    Randomly choose 500 samples from the training set and the number of valid words of each  is less than or equal to 100
    :return:
    '''
    random_seed = 20192025
    save_path = "/home/dgl/project/pfa_extraction/data/imdb_for_dfa"
    rnd_train_pos_path = os.path.join(save_path, "pos")
    rnd_train_neg_path = os.path.join(save_path, "neg")

    source_folder = "/home/dgl/project/pfa-data-generator/data/aclImdb"
    stop_words_list_path = "/home/dgl/project/pfa-data-generator/data/stopwords.txt"
    train_pos_path = os.path.join(source_folder, 'train', 'pos')
    train_neg_path = os.path.join(source_folder, 'train', 'neg')
    train_neg_files_list = os.listdir(train_neg_path)
    train_pos_files_iist = os.listdir(train_pos_path)
    dataProcessor = IMDB_Data_Processor(None, stop_words_list_path)

    train_size = 500//2

    np.random.seed(random_seed)
    random_train_neg = np.random.choice(train_neg_files_list, 5000, replace=False)
    np.random.seed(random_seed)
    random_train_pos = np.random.choice(train_pos_files_iist, 5000, replace=False)

    # (files_list, src_folder, dest_folder, dataProcessor)
    copy_files_with_filter(random_train_pos, train_pos_path, rnd_train_pos_path, dataProcessor,train_size)
    copy_files_with_filter(random_train_neg, train_neg_path, rnd_train_neg_path, dataProcessor,train_size)

    print("Saved in {}".format(os.path.abspath("/home/dgl/project/pfa_ijcai/data/imdb_for_dfa")))

if __name__ == '__main__':
    make_data_for_DFA()

