import numpy as np
import tensorflow as tf
import model
import load_data
import visual_data
import random

EPOCH_NUM = 5000

def next_batch(x, batch_size=100):
    i = 0
    while i < len(x):
        yield x[i:i + batch_size]
        i = i + batch_size

def next_batch(x, y, batch_size=100):
    i = 0
    while(i < len(x)):
        yield x[i:i + batch_size], y[i:i + batch_size]
        i = i + batch_size

if __name__ == '__main__':
    sess = tf.Session()
    sim_network = model.Main_model()
    domain_ = tf.placeholder(tf.float32, shape=[None, 2])
    label_ = tf.placeholder(tf.float32, shape=[None, 10])

    source_data, source_label = load_data.load_USPS_data()
    target_data_ld, targat_label_ld = load_data.load_MNIST_data_labeled()
    target_data_uld, _ = load_data.load_MNIST_data_unlabeled()
    
    # Train the domain classifier and the confusion loss
    # get data
    domain_label_1 = np.array([[1, 0] for i in range(len(source_data))])
    domain_label_2 = np.array([[0, 1] for i in range(len(target_data_ld) + len(target_data_uld))])
    domain_label = np.concatenate((domain_label_1, domain_label_2))
    domain_data = np.concatenate((source_data, target_data_ld, target_data_uld))
    
    random_index = [i for i in range(len(domain_label))]
    random.shuffle(random_index)
    
    domain_label = domain_label[random_index]
    domain_data = domain_data[random_index]
    
    domain_train_step = tf.train.AdamOptimizer(1e-4).minimize(sim_network.domain_loss(label=domain_))
    domain_correct_prediction = tf.equal(tf.argmax(domain_, 1), tf.argmax(sim_network.h_fcD, 1))
    domain_accuracy = tf.reduce_mean(tf.cast(domain_correct_prediction, tf.float32))
    
    confusion_train_step = tf.train.AdamOptimizer(1e-4).minimize(sim_network.domain_loss(label=domain_))
    
    classifier_train_step = tf.train.AdamOptimizer(1e-4).minimize(sim_network.classifier_loss(label=label_))
    classifier_correct_prediction = tf.equal(tf.argmax(label_, 1), tf.argmax(sim_network.h_fc8, 1))
    classifier_accuracy = tf.reduce_mean(tf.cast(classifier_correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    print("Training the domain classifier and the confusion loss")
    for epoch_index in range(EPOCH_NUM):
        batch_index = 0
        for image, domain in next_batch(domain_data, domain_label, batch_size=100000):
            print("epoch %d, batch %d" % (epoch_index, batch_index))
            batch_index = batch_index + 1
            
            sess.run(domain_train_step, feed_dict={sim_network.input_data:image, domain_:domain})
            
            sim_network.W_fcD.trainable = False
            sim_network.b_fcD.trainable = False
            confusion = np.array([[0.5, 0.5] for i in range(len(image))])
            # print(confusion)
            # print(domain)
            
            sess.run(confusion_train_step, feed_dict={sim_network.input_data:image, domain_:confusion})
            sim_network.W_fcD.trainable = True
            sim_network.b_fcD.trainable = True

        accuracy = domain_accuracy.eval(feed_dict={sim_network.image_input:image, domain_:domain})
        print("Training domain accuracy %g" % accuracy)
    
    print(domain_label)
    # classifier_loss = sim_network.classifier_loss()
    # classifier_train_step = tf.train.AdamOptimizer(1e-4).minimize(sim_network.)