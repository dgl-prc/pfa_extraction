
class Vob():
    def __init__(self):
        self.vocabulary=set()
        self.salience_map={} #trans label count

    def add_word(self,words):
        if isinstance(words,list):
            for word in words:
                self.add_word(word)
                self.update_salience_map(word)
        else:
            self.vocabulary.add(words)
            self.update_salience_map(words)

    def update_salience_map(self,word):
        if word not in self.salience_map.keys():
            self.salience_map[word]={"pos":0,"neg":0,"neuter":0}


    def parse_trace(self,word_sequence,label_trace):
        last_label = -1
        for word,label in zip(word_sequence,label_trace):
            if label == last_label:
                self.salience_map[word]["neuter"] += 1
            else:
                if label == 1:
                    self.salience_map[word]["pos"]+=1
                if label == 0:
                    self.salience_map[word]["neg"] += 1
            last_label = label

    def parse_trace_without_neuter(self,word_sequence,label_trace):

        for word,label in zip(word_sequence,label_trace):
                if label == 1:
                    self.salience_map[word]["pos"]+=1
                elif label == 0:
                    self.salience_map[word]["neg"] += 1
                else:
                    continue



