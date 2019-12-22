import numpy as np

def make_wv_matrix(data, word_vectors):
    wv_matrix = []
    for i in range(len(data["vocab"])):
        word = data["idx_to_word"][i]
        if word in word_vectors.vocab:
            wv_matrix.append(word_vectors.word_vec(word))
        else:
            wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
    # one for UNK and one for zero padding
    wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
    wv_matrix.append(np.zeros(300).astype("float32"))
    wv_matrix = np.array(wv_matrix)
    return wv_matrix

def set_data(x, y, test_idx):
    data = {}
    data["train_x"], data["train_y"] = x[:test_idx], y[:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]
    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
    return data
