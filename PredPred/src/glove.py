import torch


class Glove:
    def __init__(self, glove_file_path):
        f = open(glove_file_path, "r", encoding="utf-8")
        self.word_to_idx = {}
        self.mat = []
        for i, line in enumerate(f):
            word, *vec = line.split()
            self.word_to_idx[word] = i
            self.mat.append(torch.tensor(
                [float(x) for x in vec],
                dtype=torch.float)
            )

        f.close()

    def get(self, word):
        return self.mat[self.word_to_idx.get(word)]

    def has(self, word):
        return word in self.word_to_idx
