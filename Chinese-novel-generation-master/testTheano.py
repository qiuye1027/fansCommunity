import helper
import re
from theano import function, config, shared, tensor, sandbox
import numpy
import time

# 读入数据
dir = './data/寒门首辅.txt'
text = helper.load_text(dir)

# 设置一下要用多少个字来训练，方便调试。这里先用100000字进行训练
num_words_for_training = 100000
text = text[:num_words_for_training]

lines_of_text = text.split('\n')
lines_of_text = lines_of_text[14:]
lines_of_text = [lines for lines in lines_of_text if len(lines) > 0]
# print(lines_of_text[:20])
lines_of_text = [lines.strip() for lines in lines_of_text]

# 生成一个正则，负责找『[]』包含的内容
pattern = re.compile(r'\[.*\]')
# 将所有指定内容替换成空
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]
# print(lines_of_text[:20])
# 将上面的正则换成负责找『<>』包含的内容
pattern = re.compile(r'<.*>')

# 将所有指定内容替换成空
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]

# 下一步，把每句话最后的『……』换成『。』。
# 将上面的正则换成负责找『......』包含的内容
pattern = re.compile(r'\.+')

# 将所有指定内容替换成空
lines_of_text = [pattern.sub("。", lines) for lines in lines_of_text]

# 将上面的正则换成负责找行中的空格
pattern = re.compile(r' +')

# 将所有指定内容替换成空
lines_of_text = [pattern.sub("，", lines) for lines in lines_of_text]


# 将上面的正则换成负责找句尾『\\r』的内容
pattern = re.compile(r'\\r')

# 将所有指定内容替换成空
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]


# 创建文字对应数字和数字对应文字的两个字典
def create_lookup_tables(input_data):

    vocab = set(input_data) #一个无序不重复元素集


    # 文字到数字的映射
    vocab_to_int = {word: idx for idx, word in enumerate(vocab)}

    # 数字到文字的映射
    int_to_vocab = dict(enumerate(vocab))

    return vocab_to_int, int_to_vocab

# 创建一个符号查询表,把逗号，句号等符号与一个标志一一对应，用于将『我。』和『我』这样的类似情况区分开来，排除标点符号的影响。
def token_lookup():

    symbols = set(['。', '，', '“', "”", '；', '！', '？', '（', '）', '——', '\n'])

    tokens = ["P", "C", "Q", "T", "S", "E", "M", "I", "O", "D", "R"]

    return dict(zip(symbols, tokens))

# 预处理一下数据，并保存到磁盘，以便下次直接读取  ================================================================================================================
helper.preprocess_and_save_data(''.join(lines_of_text), token_lookup, create_lookup_tables)

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

# 训练循环次数
num_epochs = 200

# batch大小
batch_size = 256

# lstm层中包含的unit个数
rnn_size = 512

# embedding layer的大小
embed_dim = 512

# 训练步长
seq_length = 30

# 学习率
learning_rate = 0.003

# 每多少步打印一次训练信息
show_every_n_batches = 30

# 保存session状态的位置
save_dir = './save'
