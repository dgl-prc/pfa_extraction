import numpy as np
class Classifier(object):
    def __init__(self, rnn, dataProcessor):
        self.cuda_no = 0
        self.rnn = rnn.cuda(self.cuda_no)
        self.dataProcessor = dataProcessor

    def get_probs(self, sent):
        '''
        :param sent: the word list
        :return:
        '''
        tensor_sequence = self.dataProcessor.sequence2tensor(sent, 300)
        if self.cuda_no != -1:
            tensor_sequence = tensor_sequence.cuda(self.cuda_no)
        output, hn = self.rnn(tensor_sequence)
        probs = self.rnn.output_pr_dstr(hn[-1]).cpu().detach().numpy()
        return probs

    def get_label(self, sent):
        '''
        :param sent: the sentence, word list
        :return:
        '''
        probs = self.get_probs(sent)
        return np.argmax(probs, 1)[0]