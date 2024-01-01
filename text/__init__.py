import os
from g2p_en import G2p
import torch
from tokenizers import Tokenizer

from text.cleaners import english_cleaners
from text.symbols import symbols

DEFAULT_VOCAB_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/tokenizer.json')

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence([s for s in text.split()])


def sequence_to_text(sequence):
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            result += s
    return result


def text_to_sequence(text):
    sequence = []
    g2p = G2p()
    out = g2p(text)

    for word in out:
        if word == ' ':
            sequence += _symbols_to_sequence(' ')
        elif word == '?' or word == '.' or word == '!' or word == ';' or word == ',' or word == ':':
                sequence = sequence[:-1]
                sequence += _arpabet_to_sequence(word)
        else:
            sequence += _arpabet_to_sequence(word)

    return sequence


class VoiceBpeTokenizer:
    def __init__(self, vocab_file=None):
        print('VoiceBpeTokenizer initialized...')
        self.tokenizer = Tokenizer.from_file(
            DEFAULT_VOCAB_FILE if vocab_file is None else vocab_file
        )
        self.preprocess_text = english_cleaners

    def encode_text(self, txt):
        txt = self.preprocess_text(txt)
        return txt

    def encode(self, txt):
        txt = self.preprocess_text(txt)
        txt = txt.replace(' ', '[SPACE]')
        ids = self.tokenizer.encode(txt).ids
        # print(ids)
        # print(txt)
        # print(self.decode(ids))
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(' ', '')
        txt = txt.replace('[SPACE]', ' ')
        txt = txt.replace('[STOP]', '')
        txt = txt.replace('[UNK]', '')
        return txt

    def symbols(self):
        vocab = self.tokenizer.get_vocab()
        symbols = sorted(vocab, key=lambda x: vocab[x])
        return symbols

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


class ArpaTokenizer:
    def __init__(self):
        print('ArpaTokenizer initialized...')
        self.preprocess_text = english_cleaners

    def encode(self, txt):
        txt = self.preprocess_text(txt)
        ids = text_to_sequence(txt)
        # print(ids)
        # print(txt)
        # print(self.decode(ids))
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = sequence_to_text(seq)
        return txt

    def vocab_size(self):
        return len(symbols)



if __name__ == "__main__":
    tokenizer = ArpaTokenizer()
    print(tokenizer.vocab_size())