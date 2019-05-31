# just for fun.
# by zhouke
# 导入tensorflow
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


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    print("input_tensor size :",input_tensor.shape)
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_= tf.placeholder(tf.int32,[None,OUTPUT_NODE],name='y-input')
    global_step = tf.Variable(0,trainable=False)
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev = 0.1))
    biases1  = tf.Variable(tf.constant(0.1,shape = [LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev = 0.1))
    biases2  = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))
    y = inference(x,None,weights1,biases1,weights2,biases2)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_average_op = variable_average.apply(tf.trainable_variables())
    average_y = inference(x,variable_average,weights1,biases1,weights2,biases2)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variables_average_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed={x:mnist.test.images,y_:mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0 :
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy "
                    "using average model is %g"%(i,validate_acc))
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average"
            "model is %g"%(TRAINING_STEPS,test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("./mnist_data/",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

