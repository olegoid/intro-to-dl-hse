import os
import torch
from typing import Union, List, Tuple
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tokenizers import BertWordPieceTokenizer

class UnigramTextDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42
    VAL_RATIO = 0.05

    def __init__(self, data_file: str, train: bool = True, tokenizer_path: str = None, vocab_size: int = 2000, max_length: int = 128):
        """
        Dataset with texts, supporting unigram tokenizer
        :param data_file: txt file containing texts
        :param train: whether to use train or validation split
        :param tokenizer_path: path to save the tokenizer model
        :param vocab_size: tokenizer vocabulary size
        :param max_length: maximal length of text in tokens
        """
        if not os.path.isfile(tokenizer_path + '.json'):
            # Train the tokenizer if not trained yet
            tokenizer = BertWordPieceTokenizer(lowercase=False)
            tokenizer.train(files=[data_file], vocab_size=vocab_size, min_frequency=1)
            tokenizer.save(tokenizer_path)

        self.tokenizer = BertWordPieceTokenizer(tokenizer_path)
        self.texts = open(data_file).readlines()

        # Split texts into train and validation
        train_texts, val_texts = train_test_split(self.texts, test_size=self.VAL_RATIO, random_state=self.TRAIN_VAL_RANDOM_SEED)

        self.texts = train_texts if train else val_texts
        self.max_length = max_length
        self.vocab_size = len(self.tokenizer.get_vocab())

    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """ Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return [self.tokenizer.encode(text).ids for text in texts]

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]) -> Union[str, List[str]]:
        """ Decode indices as text or list of texts
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()
        return [self.tokenizer.decode(ids) for ids in ids]

    def __len__(self):
        """ Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.texts)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        """ Tokenize the text and pad to the maximal length
        :param item: text id
        :return: encoded text indices and its actual length
        """
        text = self.texts[item]
        encoding = self.tokenizer.encode(text)

        input_ids = encoding.ids
        length = len(input_ids)

        if length > self.max_length:
            input_ids = input_ids[:self.max_length]
            length = self.max_length

        padding_length = self.max_length - length
        input_ids += [0] * padding_length  # 0 represents [PAD]

        return torch.tensor(input_ids), length