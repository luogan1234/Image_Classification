from Data import Data
import tensorflow as tf
import scipy.io as scio


class Data:
    def __init__(self, name):
        self.name = name
        self.feature_train = []
        self.label_train = []
        self.feature_validation = []
        self.label_validation = []
        self.feature_test = scio.loadmat('../data/' + name + '_test.mat')[name + '_test']
        train = scio.loadmat('../data/' + name + '_train.mat')[name + '_train']
        validation = scio.loadmat('../data/' + name + '_validation.mat')[name + '_validation']
        print 'read '+self.name+' data done.'
        for vec in train:
            self.feature_train.append(vec[:-1])
            self.label_train.append(vec[-1])
        for vec in validation:
            self.feature_validation.append(vec[:-1])
            self.label_validation.append(vec[-1])
        for i in range(len(self.label_train)):
            if self.label_train[i] == 1.0:
                self.label_train[i] = [0.0, 1.0]
            else:
                self.label_train[i] = [1.0, 0.0]
        for i in range(len(self.label_validation)):
            if self.label_validation[i] == 1.0:
                self.label_validation[i] = [0.0, 1.0]
            else:
                self.label_validation[i] = [1.0, 0.0]
        


class Alex:
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(tf.float32, [None, 4096])
            self.label = tf.placeholder(tf.float32, [None, 2])
            self.w1 = tf.Variable(tf.random_normal([4096, 4096], stddev = 0.1))
            self.b1 = tf.Variable(tf.random_normal([1, 4096], stddev = 0.1))
            self.w2 = tf.Variable(tf.random_normal([4096, 2], stddev = 0.1))
            self.b2 = tf.Variable(tf.random_normal([1, 2], stddev = 0.1))
            self.istest = tf.placeholder(tf.bool)
            self.iter = tf.placeholder(tf.int32)
            self.fc1 = tf.matmul(self.input, self.w1)
            self.fc1bn, self.fc1ema = self.batchnorm(self.fc1, self.istest, self.iter, self.b1)
            self.relu1 = tf.nn.relu(self.fc1)
            self.fc2 = tf.matmul(self.relu1, self.w2) + self.b2
            self.predict = tf.argmax(self.fc2, axis = 1)
            self.ground = tf.argmax(self.label, axis = 1)
            self.score = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.ground), tf.float32))
            self.loss = tf.losses.softmax_cross_entropy(self.label, self.fc2) + 0.001 * (tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2))
            self.netloss = tf.losses.softmax_cross_entropy(self.label, self.fc2)
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
            self.trainstep = [self.optimizer, self.fc1ema]
            self.init = tf.global_variables_initializer()

    def batchnorm(self, x, istest, iteration, offset):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        mean, variance = tf.nn.moments(x, [0])
        update_moving_average = exp_moving_avg.apply([mean, variance])
        m = tf.cond(istest, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(istest, lambda: exp_moving_avg.average(variance), lambda: variance)
        xbn = tf.nn.batch_normalization(x, m, v, offset, None, 1e-5)
        return xbn, update_moving_average



if __name__ == '__main__':
    data = Data('cat')
    model = Alex()
    with model.graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            sess.run(model.init)
            for i in range(10000):
                feed = {model.input: data.feature_train, model.label: data.label_train, model.istest: False, model.iter: i}
                sess.run(model.trainstep, feed_dict = feed)
                feed = {model.input: data.feature_train, model.label: data.label_train, model.istest: True, model.iter: i}
                trainloss = sess.run(model.loss, feed_dict = feed)
                trainnetloss = sess.run(model.netloss, feed_dict = feed)
                feed = {model.input: data.feature_validation, model.label: data.label_validation, model.istest: True, model.iter: i}
                validationloss = sess.run(model.loss, feed_dict = feed)
                validationnetloss = sess.run(model.netloss, feed_dict = feed)
                score = sess.run(model.score, feed_dict = feed)
                print(i, trainloss, trainnetloss, validationloss, validationnetloss, score)
