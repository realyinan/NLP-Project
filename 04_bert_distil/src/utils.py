import torch
from tqdm import tqdm
import time
from datetime import timedelta
import os
import pickle as pkl


UNK, PAD, CLS = "[UNK]", "[PAD]", "[CLS]"  # padding符号, bert中综合信息符号
MAX_VOCAB_SIZE = 10000  # 词表长度限制

def build_vocab(file_path, tokenizer, max_size, min_freq):
    """
    :param file_path: 包含文本数据的文件路径
    :param tokenizer: 用于分词的函数
    :param max_size: 词汇表的最大大小
    :param min_freq: 词汇表中的最小出现频率, 低于此频率的词汇将被过滤掉
    :return: vocab_dict 一个字典, 将词汇映射到索引词汇表
    """
    vocab_dic = {}  # 用于存储词汇表的字典, 键为单词, 值为单词出现的次数
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content = line.split("\t")[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        # 根据词频对词汇表进行排序, 并选择出频率较高的词汇
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] > min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        # 将选定的词汇构建为字典, 键为单词, 值为索引
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic)+1, CLS: len(vocab_dic)+2})
        return vocab_dic


def build_dataset_cnn(config):
    tokenizer = lambda x: [y for y in x]  # 定义字符级别的分词器
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, "rb"))
    else:
        # 构建词汇表
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        # 保存词汇表
        pkl.dump(vocab, open(config.vocab_path, "wb"))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []  # 用于存储处理后的数据列表
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, label = line.split("\t")
                words_line = []
                token = tokenizer(content)  # 使用分词器对内容进行分词
                seq_len = len(token)

                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                       token = token[:pad_size]
                       seq_len = pad_size
                # 将词转换为对应的id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents

    # 加载训练和测试数据集
    train = load_dataset(config.train_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return vocab, train, test


def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []  # 用于存储处理后的数据列表
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, label = line.split("\t")
                token = config.tokenizer.tokenize(content)  # 使用分词器对内容进行分词
                token = [CLS] + token
                seq_len = len(token)
                mask = []  # 用于存储填充掩码
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size-len(token))
                        token_ids += [0] * (pad_size - len(token))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    # 使用load_dataset函数加载训练集, 测试集, 验证集
    train = load_dataset(config.train_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return train, test

class DatasetIterator(object):
    def __init__(self, batches, batch_size, device, model_name):
        self.batch_size = batch_size
        self.batches = batches  # 包含样本的列表
        self.model_name = model_name
        self.n_batches = len(batches) // batch_size  # 批次的数量
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        """
        将样本数据转换为tensor
        """
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        # 若为BERT模型, 返回
        if self.model_name == "bert":
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y
        # 若为TextCNN模型, 返回
        elif self.model_name == "textCNN":
            return (x, seq_len), y
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def __next__(self):
        """
        获取下一个批次样本
        """
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        """
        返回迭代器本身
        """
        return self

    def __len__(self):
        """
        获取迭代器长度
        """
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    """
    根据配置信息构建迭代器
    """
    iterator = DatasetIterator(dataset, config.batch_size, config.device, config.model_name)
    return iterator


def get_time_diff(start_time):
    """
    计算已经使用的时间
    """
    end_time = time.time()
    # 计算时间差
    time_diff = end_time - start_time
    # 将是时间差转换为整数秒, 返回一个时间差对象
    return timedelta(seconds=int(round(time_diff)))