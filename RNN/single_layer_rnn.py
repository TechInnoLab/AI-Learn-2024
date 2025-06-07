import random  # 导入随机数模块
from torch.nn import functional as F  # 导入PyTorch的函数式API
from torch import nn  # 导入PyTorch的神经网络模块
import torch  # 导入PyTorch主库
import math  # 导入数学库
import collections  # 导入集合模块，用于计数
import string  # 导入字符串处理模块
import swanlab  # 导入swanlab用于实验日志记录
import time  # 导入时间模块


# 替换文本中的所有标点符号为空格
def replace_punctuation_with_space(text):
    punctuation = string.punctuation + "。，、？！：；“”‘’（）【】《》—……"
    translator = str.maketrans(punctuation, " " * len(punctuation))
    return text.translate(translator)


# 读取文本文件并去除标点
def read_time_machine():
    with open('Ring.txt', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    return [replace_punctuation_with_space(line).strip() for line in lines]


lines = read_time_machine()  # 读取并处理文本


# 将每一行文本分割为字符列表
def tokenize(lines):
    return [list(line) for line in lines]


tokens = tokenize(lines)  # 得到分词后的结果


# 词表类，负责token到索引的映射
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)  # 统计词频
        self._token_freqs = sorted(
            counter.items(), key=lambda x: x[1], reverse=True)  # 按频率排序
        self.idx_to_token = ['<unk>'] + reserved_tokens  # 索引到token的映射，0为未知
        self.token_to_idx = {token: idx for idx, token in enumerate(
            self.idx_to_token)}  # token到索引的映射
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)  # 单个token
        return [self.__getitem__(token) for token in tokens]  # 多个token

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]  # 单个索引
        return [self.idx_to_token[index] for index in indices]  # 多个索引

    @property
    def unk(self):
        return 0  # 未知token的索引

    @property
    def token_freqs(self):
        return self._token_freqs  # 词频列表


# 统计token出现次数
def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# 加载语料库，返回token索引序列和词表
def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines)
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# 随机采样小批量序列数据
def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]  # 随机偏移
    num_subseqs = (len(corpus) - 1) // num_steps  # 可采样的子序列数
    initial_indices = list(
        range(0, num_subseqs * num_steps, num_steps))  # 每个子序列的起始索引
    random.shuffle(initial_indices)  # 打乱顺序

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size  # 批次数
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


# 顺序采样小批量序列数据
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)  # 随机偏移
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * \
        batch_size  # 可用token数
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


# 数据加载器，支持随机和顺序采样
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


# 加载数据和词表
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=-1):
    data_iter = SeqDataLoader(batch_size, num_steps,
                              use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


batch_size, num_steps = 32, 8  # 批量大小和序列长度
train_iter, vocab = load_data_time_machine(batch_size, num_steps)  # 获取训练数据和词表


# 初始化RNN参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))  # 输入到隐藏层权重
    W_hh = normal((num_hiddens, num_hiddens))  # 隐藏到隐藏层权重
    b_h = torch.zeros(num_hiddens, device=device)  # 隐藏层偏置
    W_hq = normal((num_hiddens, num_outputs))  # 隐藏到输出层权重
    b_q = torch.zeros(num_outputs, device=device)  # 输出层偏置
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# 初始化RNN隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


# 单层RNN的前向传播
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)  # 计算隐藏状态
        Y = torch.mm(H, W_hq) + b_q  # 计算输出
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


# 从零实现的RNN模型
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)  # 独热编码
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# 文本生成函数，给定前缀生成后续字符
def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]  # 以第一个字符为起点

    def get_input(): return torch.tensor(
        [outputs[-1]], device=device).reshape((1, 1))

    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# 梯度裁剪，防止梯度爆炸
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# 单个epoch的训练过程
def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    state = None
    total_loss, total_num = 0.0, 0
    start_time = time.time()
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        total_loss += l.item() * y.numel()
        total_num += y.numel()
    elapsed = time.time() - start_time
    return math.exp(total_loss / total_num), total_num / elapsed


# 训练模型主函数
def train_model(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    with open("loss.txt", 'w') as f:
        f.write('')
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        def updater(batch_size): return sgd(net.params, lr, batch_size)

    def predict(prefix): return predict_ch8(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(
            net, train_iter, loss, updater, device, use_random_iter)
        with open("loss.txt", 'a') as f:
            f.write(f"{ppl}\n")
        swanlab.log({"perplexity": ppl})
    torch.save(net, 'rnn_model.pth')
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('交界地'))
    print(predict('莱德厄斯'))
    swanlab.finish()


# 运行训练或预测
def model_run(train=True):
    if train:
        num_epochs, lr = 200, 1  # 训练轮数和学习率
        swanlab.init(
            project="RNN-zh",
            config={
                "learning_rate": lr,
                "architecture": "RNN",
                "dataset": "Ring",
                "epochs": num_epochs
            }
        )
        num_hiddens = 512  # 隐藏层单元数
        net = RNNModelScratch(len(vocab), num_hiddens,
                              try_gpu(), get_params, init_rnn_state, rnn)
        train_model(net, train_iter, vocab, lr, num_epochs, try_gpu())
    else:
        net = torch.load("rnn_model.pth")  # 加载已训练模型
        prefix = input("请输入：")  # 输入前缀
        result = predict_ch8(prefix, 50, net, vocab, try_gpu())
        print(result)


# 检测是否有可用GPU
def try_gpu():
    """检测是否有可用GPU，返回torch.device对象"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 小批量随机梯度下降优化器
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == "__main__":
    model_run(train=False)  # 将train设置为True以训练模型，否则加载已训练的模型进行预测
