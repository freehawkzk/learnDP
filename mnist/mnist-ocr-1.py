# just for fun.
# by zhouke
# 导入tensorflow
import os
import tensorflow as tf
# 导入mnist数据集
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关数据
# 输入层节点数，由于mnist数据集图像是1*784=1*(28*28)的输入，所以输入层节点数是784
INPUT_NODE = 784 

# 输出层节点数，由于预测手写数字是0-9共10个数字，因此，输出层是10个。
# 这里输出层的数目等于需要分的类别数目。
OUTPUT_NODE = 10 
                 
# 隐层节点数。这里使用只有一个隐层，500个节点。
LAYER1_NODE = 500

# 一个训练batch中的样本数，数字越小，训练过程越接近随机梯度下降，数字越大，越接近梯度下降
BATCH_SIZE = 100 

#基础学习率
LEARNING_RATE_BASE = 0.8 

# 学习率的衰减率
LEARNING_RATE_DECAY = 0.99 

# 描述模型复杂度的正则化项在损失函数中的系数
REGULARIZATION_RATE = 0.0001 

# 总训练轮数
TRAINING_STEPS = 30000 

# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99 

# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。在这里定义了一个使用RELU激活函数
# 的三层全连接神经网络。通过加入隐藏层实现了多层网络结构，通过ReLU激活函数实现了去线性化。在这个函数中也支持
# 传入用于计算参数平均值的类，这样方便在测试时使用滑动平均模型
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐层的前向传播结果，这里使用了ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        
        # 计算输出层的前向传播结果。
        # 1. 因为在计算损失函数的时候会一并计算softmax函数，因此这里不需要加入激活函数。
        # 2. 不加入softmax对预测结果没有影响，因为预测时使用的时不同类别对应节点输出值的相对大小，有没有
        #   softmax层对最后的分类结果计算没有影响。
        # 于是在计算整个神经网络的前向传播时可以不加入最后的softmax层。
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 提供了滑动平均类时，首先使用avg_class.average函数来计算出变量的滑动平均值，
        # 然后用于计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# 训练模型的过程
# 网络中各层的输入：
#       输入层：    1          * INPUT_NODE
#       隐层权重:   INPUT_NODE * LAYER1_NODE
#       隐层偏置：  1          * LAYER1_NODE
#       输出层权重：LAYER1_NODE*OUTPUT_NODE
#       输出层偏置：  1        * OUTPUT_NODE
def train(mnist):

    # 输入变量的占位符
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    
    # 输出变量的占位符
    y_= tf.placeholder(tf.int32,[None,OUTPUT_NODE],name='y-input')
    
    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量，将trainable标志置为False。
    # 在tf中，一般将代表训练轮数的变量指定为不可训练的变量。
    global_step = tf.Variable(0,trainable=False)

    # 生成隐层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev = 0.1))
    biases1  = tf.Variable(tf.constant(0.1,shape = [LAYER1_NODE]))
    
    # 输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev = 0.1))
    biases2  = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))
    
    # 计算在当前参数下神经网络的前向传播结果。这里未指定滑动平均类，因此不执行滑动平均过程。
    y = inference(x,None,weights1,biases1,weights2,biases2)

    # 给定滑动平均衰减率和训练轮数，初始化滑动平均类。给定训练轮数的变量可以加快训练早期变量的更新速度
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量，比如global_step，不需要使用滑动平均。
    # tf.trainable_variables返回的是图上集合GraphKeys.TRAINABLE_VARIABLES中的元素，这个集合中
    # 的元素是所有没有指定trainable=Flase的参数。
    variables_average_op = variable_average.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果。滑动平均不会改变变量本身的取值，而是会维护一个影子变量来记录
    # 其滑动平均值。当需要使用这个滑动平均值时，需要明确调用average函数
    average_y = inference(x,variable_average,weights1,biases1,weights2,biases2)

    # 计算交叉熵作为刻画预测值和真实值之间的损失函数。这里使用tf中提供的sparse_softmax_cross_entropy_with_logits
    # 函数来计算交叉熵。当分类问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。MNIST问题中，图片
    # 只包含0-9中的一个数字，所以可以使用交叉熵函数来计算交叉熵损失。
    # sparse_softmax_cross_entropy_with_logits参数logits是网络计算的预测结果，labels是训练样本给定的标签
    # 由于标准答案是一个长度为10的一维数组，而sparse_softmax_cross_entropy_with_logits函数需要提供一个正确
    # 答案的数字，因此，需要使用tf.argmax函数来得到标准答案中对应的类别的编号。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y)

    # 计算在当前batch中所有训练样本的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失。一般只计算神经网络上权重的正则化损失，而不计算偏置项
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, # 基础学习率，
                                                                   # 随着迭代的进行，更新变量时使用的学习率在这个基础上递减
                                               global_step,# 当前迭代的轮数
                                               mnist.train.num_examples/BATCH_SIZE,# 过完所有的训练数据需要的迭代次数
                                               LEARNING_RATE_DECAY)# 学习率衰减速度
    
    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数，这里包含了交叉熵损失和正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据都需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值。
    # 为了一次完成多个操作，tensorflow提供了tf.control_dependencies和tf.group两种机制。这两行程序与
    # train_op = tf.group(train_step, variables_averages_op)等价。
    with tf.control_dependencies([train_step,variables_average_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确。tf.argmax(average_y,1)计算出每一个样本的预测答案。
    # 其中average_y是一个batch_size * 10的二维数组，每一行表示一个样本的前向传播结果。tf.argmax的第二个参数
    # “1”表示选取最大值的操作仅在第一个维度中进行，也就是说，只在每一行中选择最大值对应的下标。于是得到的结果是
    # 一个长度为batch_size的一维数组，这个一维数组中的值就表示了每一个样本对应的数字识别结果。tf.equal判断两个
    # 张量的每一维是否相等，如果相等返回True,否则返回False.
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

    # 计算预测准确率的平均值。correct_prediction是对每一个样本预测正确与否的判断，将其转换为float32，再计算
    # 均值，可以表示平均准确率。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        # 初始化所有变量
        tf.global_variables_initializer().run()
        
        # 准备验证数据，一般在神经网络中会通过验证数据来大致验证判断停止的条件和评判训练的结果
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}

        # 准备测试数据，在真实应用中，这部分数据在训练时不可见，这些数据只作为模型优劣的最后评价标准
        test_feed={x:mnist.test.images,y_:mnist.test.labels}

        # 迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0 :
                # 计算滑动平均模型在验证数据集上的结果。因为mnist数据集比较小，所以一次可以处理所有的验证数据。
                # 所以这里没有将验证数据集分成较小的batch，当神经网络模型比较复杂时或者验证数据集较大时，可能
                # 会因为batch太大而导致内存溢出的错误
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g"%(i,validate_acc))

            # 产生这一轮所用的batch的训练数据，并运行训练过程
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        # 在训练结束之后，在测试数据集上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g"%(TRAINING_STEPS,test_acc))

# 主程序入口
def main(argv=None):
    # 确定所使用的mnist数据的路径，使用本脚本所在目录的子目录mnist_data中的数据
    mnistpath = os.path.split(os.path.realpath(__file__))[0] + '/mnist_data/'
    mnist = input_data.read_data_sets(mnistpath,one_hot=True)
    train(mnist)

# Tensorflow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()

