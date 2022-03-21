import keras.models
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dense
import numpy as np

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.
        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

digits = 5
maxlen = digits + 1 + digits

chars = '0123456789+- '
ctable = CharacterTable(chars)

model = keras.models.load_model('seq2seq_model2')
model.summary()

while True:
    test_seq = input(">> ")
    if len(test_seq) > maxlen:
        continue
    if test_seq == "exit":
        break
    test_seq = test_seq + ' ' * (maxlen - len(test_seq))
    seq_array = np.zeros((1, maxlen, len(chars)), dtype=bool)
    seq_array[0] = ctable.encode(test_seq, maxlen)

    preds = np.argmax(model.predict(seq_array[np.array([0])]), axis=-1)
    guess = ctable.decode(preds[0], calc_argmax=False)
    print(guess)
