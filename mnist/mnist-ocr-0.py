# just for fun.
# by zhouke
# 导入tensorflow
import tensorflow as tf
# 导入mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据集，如果本地不存在数据的话，会自动联网下载数据集
mnist = input_data.read_data_sets("./mnist_data/",one_hot=True)

# 训练集样本数目,55000
print("Training data size:%d " %(mnist.train.num_examples))

# 验证集样本数目,5000
print("Validating data size:%d " %(mnist.validation.num_examples))

# 测试集样本数目,10000
print("Testing data size:%d " %(mnist.test.num_examples))

# 输出第一个训练样本
print("Example training data: ",(mnist.train.images[0]))

# 输出第一个训练样本的标签数据
print("Example training data label: ",(mnist.train.labels[0]))

# 一个训练batch的大小为100个样本
batch_size = 100
# 从训练集中加载一个batch
xs,ys=mnist.train.next_batch(batch_size)

# 当前训练batch的输入大小，单个样本大小是1*784，100个样本，所以是100*784.784=28*28
print("X shape ",xs.shape)

# 当前训练batch的标签大小，单个样本标签大小是1*10,100个样本，所以是100*10
print("Y shape ",ys.shape)