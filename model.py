import numpy as np
import tensorflow as tf
import cv2
import os
import load_data

def init_bias_variable(shape):
    initial = tf.constant(-0.1, shape=shape)
    return tf.Variable(initial)

def init_weight_variable(shape):
    initial = tf.random_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
          
class Main_model:
    def __init__(self, keep_rate=1.0):
        self.input_data = tf.placeholder(tf.float32, [None, 784])
        self.input_image = tf.reshape(self.input_data, [-1, 28, 28, 1])
        
        self.W_conv1 = init_weight_variable([3, 3, 1, 32])
        self.b_conv1 = init_bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(self.input_image, self.W_conv1) + self.b_conv1) 
        self.h_pool1 = max_pool_2x2(self.h_conv1)
        
        self.W_conv2 = init_weight_variable([3, 3, 32, 64])
        self.b_conv2 = init_bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2) 

        self.W_conv3 = init_weight_variable([3, 3, 64, 64])
        self.b_conv3 = init_bias_variable([64])
        self.h_conv3 = tf.nn.relu(conv2d(self.h_conv2, self.W_conv3) + self.b_conv3) 
        self.h_pool3 = max_pool_2x2(self.h_conv3)
        
        self.W_conv4 = init_weight_variable([3, 3, 64, 64])
        self.b_conv4 = init_bias_variable([64])
        self.h_conv4 = tf.nn.relu(conv2d(self.h_pool3, self.W_conv4) + self.b_conv4) 

        self.W_conv5 = init_weight_variable([3, 3, 64, 128])
        self.b_conv5 = init_bias_variable([128])
        self.h_conv5 = tf.nn.relu(conv2d(self.h_conv4, self.W_conv5) + self.b_conv5) 
        self.h_conv5_flat = tf.reshape(self.h_conv5, [-1, 7 * 7 * 128])
        
        self.W_fc6 = init_weight_variable([7 * 7 * 128, 512])
        self.b_fc6 = init_bias_variable([512])
        self.h_fc6 = tf.nn.relu(tf.matmul(self.h_conv5_flat, self.W_fc6) + self.b_fc6)
        self.h_fc6_drop = tf.nn.dropout(self.h_fc6, keep_rate)
        
        self.W_fc7 = init_weight_variable([512, 1024])
        self.b_fc7 = init_bias_variable([1024])
        self.h_fc7 = tf.nn.relu(tf.matmul(self.h_fc6_drop, self.W_fc7) + self.b_fc7)
        self.h_fc7_drop = tf.nn.dropout(self.h_fc7, keep_rate)
        
        self.W_fc8 = init_weight_variable([1024, 10])
        self.b_fc8 = init_bias_variable([10])
        self.h_fc8 = tf.nn.relu(tf.matmul(self.h_fc7_drop, self.W_fc8) + self.b_fc8)
        
        self.W_fcD = init_weight_variable([1024, 2])
        self.b_fcD = init_bias_variable([2])
        self.h_fcD = tf.nn.sigmoid(tf.matmul(self.h_fc7, self.W_fcD) + self.b_fcD)
    
    def classifier_loss(self, label, logit):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
    
    def confusion_loss(self):
        return self.classifier_loss(tf.constant(0,5, shape=[2]), self.h_fcD)
        
    def domain_loss(self, label):
        return self.classifier_loss(label=label, logit=self.h_fcD)
    # def soft_loss(self, label, temperature):
    #     l = tf.Variable(float32, shape=[10, 10])
    #     
    #     
    # def main_loss(self, label, domain_label, alpha=0.01, beta=0.1, temperature=10.0):
    #     cls_loss = self.classifier_loss(label, self.fc8)
    #     confusion_loss = self.classifier_loss(tf.constant(0,5, shape=[2]), self.fcD)
    #     soft_loss = 
    #     return cls_loss + alpha * confusion_loss + beta * soft_loss