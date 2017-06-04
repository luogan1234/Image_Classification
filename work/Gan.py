import sugartensor as tf
import numpy as np
from Data import Data

def generator(tensor):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    with tf.sg_context(name='generator', size=4, stride=2, act='leaky_relu', bn=True, reuse=reuse):
        res = (tensor
               .sg_dense(dim=1024, name='fc1')
               .sg_dense(dim=4096, act='relu', bn=False, name='fc3'))

        return res


def discriminator(tensor):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
    with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu', bn=True, reuse=reuse):
        res = (tensor
               .sg_dense(dim=4096, name='fc1')
               .sg_dense(dim=512, name='fc2')
               .sg_dense(dim=1, act='sigmoid', bn=False, name='fc3')
               .sg_squeeze())   
        return res

batch_size = 27
rand_dim = 512
raw = Data('bird')
raw.readData()
def prepareData():
    data = raw
    ret = {}
    positive = np.array(data.label_train) > 0
    # negative = np.array(data.label_train) < 0.5
    # ret['train'] = tf.sg_data._data_to_tensor([np.array(data.feature_train, dtype=np.float32)[positive]], batch_size = batch_size, name = 'train_pos')
    # ret['train_negative'] = tf.sg_data._data_to_tensor([np.array(data.feature_train, dtype=np.float32)[negative]], batch_size = batch_size, name = 'train_neg')
    all_feature = np.concatenate((np.array(data.feature_train, dtype=np.float32), np.array(data.feature_validation, dtype= np.float32)))
    all_label = np.concatenate((np.array(data.label_train, dtype = np.float32), np.array(data.label_validation, dtype=np.float32)))
    # ret['train'] = tf.sg_data._data_to_tensor([np.array(data.feature_train, dtype=np.float32), np.array(data.label_train, dtype = np.float32)], batch_size = batch_size, name = 'train')
    ret['train'] = tf.sg_data._data_to_tensor([all_feature, all_label], batch_size = batch_size, name = 'train') 
    ret['valid'] = tf.sg_data._data_to_tensor([np.array(data.feature_validation, dtype= np.float32), np.array(data.label_validation, dtype=np.float32)], batch_size = batch_size, name = 'valid')
    return ret

def trainIt():
    data = prepareData()
    x = data['train'][0]
    # x = data['train']
    z = tf.random_normal((batch_size, rand_dim))
    gen = generator(z)
    disc_real = discriminator(x) 
    disc_fake = discriminator(gen)
    loss_d_r = disc_real.sg_mse(target = data['train'][1], name = 'disc_real')
    # loss_d_r = disc_real.sg_mse(target = tf.ones(batch_size), name = 'disc_real')
    loss_d_f = disc_fake.sg_mse(target = tf.zeros(batch_size), name = 'disc_fake')
    loss_d = (loss_d_r + loss_d_f) / 2
    loss_g = disc_fake.sg_mse(target = tf.ones(batch_size), name = 'gen')
    # train_disc = tf.sg_optim(loss_d, lr=0.01, name = 'train_disc', category = 'discriminator')  # discriminator train ops
    train_disc = tf.sg_optim(loss_d_r, lr=0.01, name = 'train_disc', category = 'discriminator')
    train_gen = tf.sg_optim(loss_g, lr=0.01, category = 'generator')  # generator train ops
    @tf.sg_train_func
    def alt_train(sess, opt):
        if sess.run(tf.sg_global_step()) % 1 == 0:
            l_disc = sess.run([loss_d_r, train_disc])[0]  # training discriminator
        else:
            l_disc = sess.run(loss_d)
        # l_gen = sess.run([loss_g, train_gen])[0]  # training generator
        # print np.mean(l_gen)
        return np.mean(l_disc) #+ np.mean(l_gen)

    alt_train(log_interval=10, max_ep=25, ep_size = (1100 + 690) / batch_size, early_stop=False,
            save_dir='asset/train/gan', save_interval=10)
def genIt(name = 'bird'):
    z = tf.random_normal((batch_size, rand_dim))
    gen = generator(z)
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                      tf.sg_phase().assign(False)))
        tf.sg_restore(sess, tf.train.latest_checkpoint('asset/train/gan'), category=['generator', 'discriminator'])
        fake_features = []
        for i in range(100):
            fake_features.append(sess.run(gen))
    np.save('../data/fake_'+name+'_negative.npy', np.array(fake_features).reshape((-1,4096)))
def testIt():
    data = raw
    positive = np.array(data.label_train) > 0
    x = tf.placeholder(tf.float32, [None, 4096])
    y = tf.placeholder(tf.float32)
    disc_real = discriminator(x)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(disc_real > 0.5, "float"), y), tf.float32))
    np.set_printoptions(precision=3, suppress=True)
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                      tf.sg_phase().assign(False)))
        # restore parameters
        tf.sg_restore(sess, tf.train.latest_checkpoint('asset/train/gan'), category=['generator', 'discriminator'])
        ans = sess.run(disc_real, feed_dict={x:np.array(data.test)})
        print np.sum(ans > 0.5)
        np.save('dm_bird.npy', ans)
testIt()

#bug log:
    # runqenue closed, insufficent -> already finished ep_size eps
    # multiple grad summary -> two optim