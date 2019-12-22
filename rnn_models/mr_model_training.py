import sys

sys.path.append("../")
from rnn_models.rnn_arch.gated_rnn import *
from rnn_models import train_args
import torch.optim as optim
import torch.nn as nn
from sklearn.utils import shuffle
import copy
from os.path import abspath
from utils.save_utils import *
from utils.constant import *
from utils.time_util import *
from data.process_utils import *


def test(data, model, params, mode="test", device="cuda:0"):
    model.eval()
    if mode == "train":
        X, Y = data["train_x"], data["train_y"]
    elif mode == "test":
        X, Y = data["test_x"], data["test_y"]
    acc = 0
    for sent, c in zip(X, Y):
        input_tensor = sent2tensor(sent, data, device, params["input_size"], params["WV_MATRIX"])

        output, _ = model(input_tensor)
        avg_h = torch.mean(output, dim=1, keepdim=False)
        pred = model.h2o(avg_h)

        label = data["classes"].index(c)
        pred = np.argmax(pred.cpu().data.numpy(), axis=1)[0]
        acc += 1 if pred == label else 0

    return acc / len(X)


def train(data, params):
    if params["rnn_type"] == MTYPE_GRU:
        model = GRU(input_size=params["input_size"], num_class=params["output_size"], hidden_size=params["hidden_size"],
                    num_layers=params["num_layers"])
    elif params["rnn_type"] == MTYPE_LSTM:
        model = LSTM(input_size=params["input_size"], num_class=params["output_size"],
                     hidden_size=params["hidden_size"],
                     num_layers=params["num_layers"])
    else:
        raise Exception("Unknow rnn type:{}".format(params["rnn_type"]))

    device = "cuda:{}".format(params["GPU"]) if params["GPU"] != 1 else "cpu"
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()
    pre_test_acc = 0
    max_test_acc = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])
        i = 0
        model.train()
        for sent, c in zip(data["train_x"], data["train_y"]):
            label = [data["classes"].index(c)]
            label = torch.LongTensor(label).to(device)
            input_tensor = sent2tensor(sent, data, device, params["input_size"], params["WV_MATRIX"])
            optimizer.zero_grad()

            output, _ = model(input_tensor)
            avg_h = torch.mean(output, dim=1, keepdim=False)
            pred = model.h2o(avg_h)

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            if i % 5000 == 0:
                print("epoch:{} progress:{}/{} loss:{}".format(e + 1, i + 1, len(data["train_x"]), loss))
            i += 1
        test_acc = test(data, model, params, mode="train", device=device)
        print("epoch:", e + 1, "/ test_acc:", test_acc)

        if params["EARLY_STOPPING"] and test_acc <= pre_test_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_test_acc = test_acc

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_train_acc = test(data, model, params, mode="train", device=device)
            best_model = copy.deepcopy(model)
    print("train_acc:{0:.4f}, test acc:{1:.4f}".format(max_train_acc, max_test_acc))
    return best_model, max_train_acc, max_test_acc


def main():
    dataset = sys.argv[1]
    gpu = int(sys.argv[2])
    model_type = sys.argv[3]

    params = getattr(train_args, "args_{}_{}".format(model_type, dataset))()
    data = load_pickle(get_path(DataPath.MR.PROCESSED_DATA))
    wv_matrix = load_pickle(get_path(DataPath.MR.WV_MATRIX))
    train_args.add_data_info(data, params)
    params["WV_MATRIX"] = wv_matrix
    params["GPU"] = gpu
    params["rnn_type"] = model_type.upper()

    model, train_acc, test_acc = train(data, params)

    print("saving model...")
    save_path = "rnn_models/pretrained/{}/{}/{}".format(dataset,model_type,folder_timestamp())
    save_path = os.path.join(PROJECT_ROOT, save_path)

    save_model(model, train_acc, test_acc, abspath(save_path))
    save_readme(save_path, ["{}:{}\n".format(key, params[key]) for key in params.keys()])


if __name__ == '__main__':
    main()
