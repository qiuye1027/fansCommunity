import tensorflow as tf
from test0409Readdata import *
import numpy as np
import random
# from ../helper  import *
from tensorflow.contrib import seq2seq


def random_distribution():
    """生成一个随机的概率列"""
    b = np.random.uniform(0.0, 1.0, size=[1, vocab_size])
    return b / np.sum(b, 1)[:, None]


def sample_distribution(distribution):  # 在概率下选择
    """从假定为标准化数组的分布中抽取一个元素作为样本概率。
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution[0])):
        s += distribution[0][i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction):
    d = sample_distribution(prediction)
    re = []
    re.append(d)
    return re


# 学习率
learning_rate =1 # 1.0
# 训练步长
num_steps =60 # 35
# lstm层中包含的unit个数
hidden_size = 300
# dropout时的保留概率
keep_prob =0.8 # 1.0
lr_decay = 0.5
# batch大小
batch_size =20 # 20
# lstm层数
num_layers = 3
# 训练循环次数
max_epoch =1000# 14


# 序列化 处理训练数据
x, y, id_to_word = dataproducer(batch_size, num_steps)
vocab_size = len(id_to_word)

size = hidden_size

# 基础的LSTM循环网络单元
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.5)
# 网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作
lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
# 构建多个独立的串联循环网络结构，产生num_layers个独立的LSTM网络结构操作列表
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell], num_layers)

# 构造多层LSTM前,使用zero_state函数构建全零初始输出特征和状态特征
initial_state = cell.zero_state(batch_size, tf.float32)
state = initial_state
# 获取已存在的变量（要求不仅名字，而且初始化方法等各个参数都一样），如果不存在，就新建一个。
embedding = tf.get_variable('embedding', [vocab_size, size])
input_data = x
targets = y

# 形参，用于定义过程，在执行的时候再赋具体的值
test_input = tf.placeholder(tf.int32, shape=[1])
test_initial_state = cell.zero_state(1, tf.float32)

# 从随机产生的初始词向量集合embedding中获取对应的样本特征
inputs = tf.nn.embedding_lookup(embedding, input_data)
test_inputs = tf.nn.embedding_lookup(embedding, test_input)

outputs = []
initializer = tf.random_uniform_initializer(-0.1, 0.1)

# 通过变量名 Model 获取变量
# 当reuse为False或者None时（这也是默认值），同一个tf.variable_scope下面的变量名不能相同

with tf.variable_scope("Model", reuse=None, initializer=initializer):
    with tf.variable_scope("r", reuse=None, initializer=initializer):
        # softmax_w、softmax_b是尺寸 [size, vocab_size]、 [vocab_size]的随机张量，表示线性分类模型的参数
        softmax_w = tf.get_variable('softmax_w', [size, vocab_size])
        softmax_b = tf.get_variable('softmax_b', [vocab_size])
    with tf.variable_scope("RNN", reuse=None, initializer=initializer):
        for time_step in range(num_steps):

            # 利用scope.reuse_variables()告诉TF想重复利用RNN的参数
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            # 实现对LSTM网络的1次循环调用
            (cell_output, state) = cell(inputs[:, time_step, :], state, )
            outputs.append(cell_output)
        # 调整矩阵维度  outputs为被调整维度的张量   [-1, size]为要调整为的形状
        output = tf.reshape(outputs, [-1, size])

        # 两个矩阵中对应元素各自相乘，表示预测的分类置信度结果
        logits = tf.matmul(output, softmax_w) + softmax_b
        # 实现损失函数的计算
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(targets, [-1])],
                                                      [tf.ones([batch_size * num_steps])])

        global_step = tf.Variable(0)
        #exponential_decay() 实现指数衰减学习率
        learning_rate = tf.train.exponential_decay( 10.0, global_step, 5000, 0.1, staircase=True)
        # 对所有步骤中的所有变量使用恒定的学习率
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # 计算梯度
        gradients, v = zip(*optimizer.compute_gradients(loss))
        # clip_by_global_norm是梯度缩放输入是所有trainable向量的梯度，和所有trainable向量，返回第一个clip好的梯度，第二个globalnorm
        # 实现梯度剪裁防止梯度爆炸
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        # 使用计算得到的梯度来更新对应的variable
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

        cost = tf.reduce_sum(loss) / batch_size
        # predict:
        teststate = test_initial_state
        (celloutput, teststate) = cell(test_inputs, teststate)
        partial_logits = tf.matmul(celloutput, softmax_w) + softmax_b
        partial_logits = tf.nn.softmax(partial_logits)

# 保存模型参数和Summary
sv = tf.train.Supervisor(logdir=None)

with sv.managed_session() as session:

    costs = 0
    iters = 0
    for i in range(max_epoch):
        _, l = session.run([optimizer, cost])
        costs += l
        iters += num_steps
        perplextity = np.exp(costs / iters)
        if i % 20 == 0:
            print(perplextity)

        if i % 100 == 0:
            p = random_distribution()
            b = sample(p)
            sentence = id_to_word[b[0]]
            for j in range(200):
                test_output = session.run(partial_logits, feed_dict={test_input: b})
                b = sample(test_output)
                sentence += id_to_word[b[0]]
            print(sentence)

    writer = tf.summary.FileWriter("logs", tf.get_default_graph())
    writer.close()