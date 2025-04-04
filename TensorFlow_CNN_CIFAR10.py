# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 22:46:10 2019

@author: aunko
"""


'''
    Deep Learning Beispiel mit purem TensorFlow, OHNE KERAS!

    CIFAR10 besteht aus 60000 Farbbildern mit je 32 x 32 Pixeln
    Labels: Flugzeuge, Autos, Vögel, Katzen, Hirsche, Hunde, Frösche, Pferde, Schiffe und Lastwagen

    Die Dateein mit Namen nach dem Muster data_batch_X enthalten serialisierte Anlerndaten 
    und test_batch ist eine ähnlich serialisierte Datei mit den Testdaten.
    
    Die Datei batches_meta enthält die Zuordnung von numerischen zu semnatischen Markierungen.

'''

'''
    Zuerst erstellen wir eine Datenverwaltung
'''

#import tensorflow as tf
import tensorboard as tensorboard
import numpy as np
import matplotlib.pyplot as plt
import os as os
import pickle as pickle
from datetime import datetime


""" ACHTUNG, dieses Beispiel ist mit TensorFlow 2.0 erstellt, jedoch läuft im Kompatibilitätsmodus für TensorFlow 1.0 """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

DATA_DIR = "DATA"
LOG_DIR = "\\LOG\\CNN_CIFAR10"
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

'''
    Liefert ein Dictionary mit den Feldern data (Bilder) und labels.
'''
def unpickle(file):
    
    with open(os.path.join(DATA_DIR, file), 'rb') as fo:
        
        dict = pickle.load(fo, encoding = 'bytes')
        
        return dict
    
'''
    Wandelt die Markierungen (Labels) von ganzen Zahlen in Vektoren der Länge 10 um,
    die an der Position der jeweiligen Zahl eine Eins, sonder Nullen enthalten.
'''
def one_hot(vec, vals = 10):
    
    n = len(vec)
    out = np.zeros((n, vals))
    
    out[range(n), vec] = 1
    
    return out


class CifarLoader(object):
    
    def __init__(self, source_files):
        
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None
        
    def load(self):
        
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d[b"data"] for d in data])
        n = len(images)
        
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float) / 255
        self.labels = one_hot(np.hstack([d[b"labels"] for d in data]), 10)
    
        return self
    
    def next_batch(self, batch_size):
        
        x, y = self.images[self._i:self._i + batch_size], self.labels[self._i:self._i + batch_size]
        
        self._i = (self._i + batch_size) % len(self.images)
        
        return x, y
    
class CifarDataManager(object):
    
    def __init__(self):
        
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)]).load()
        
        self.test = CifarLoader(["test_batch"]).load()
        
    def display_cifar(self, images, size):
        
        n = len(images)
        plt.figure(figsize = (15, 15))
        plt.gca().set_axis_off()
        
        im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) for j in range(size)])
        
        plt.imshow(im)
        plt.show()
        

d = CifarDataManager()

print("Anzahl der Bilder zum Anlernen: {}".format(len(d.train.images)))
print("Anzahl der Markierungen zum Anlernen: {}".format(len(d.train.labels)))
print("Anzahl der Bilder zum Testen: {}".format(len(d.test.images)))
print("Anzahl der Markierungen zum Testen: {}".format(len(d.test.labels)))

images = d.train.images

d.display_cifar(images, 10)

''' -------------------- Generalisierung ------------------------ '''


''' 
    Diese Funktion legt die Gewichte für entweder eine vollständig verbundene oder eine Konvolutionsschict an.
    Die Gewichte werden mit Zufallszahlen aus einer beidseitig abgeschnittenen Normalverteilung mit einer 
    Standardabweichung von 0,1 initialisiert. Diese Art der Initialisierung ist recht gebräuchlich und
    führt in der Regel zu guten Ergebnissen.
'''
def weight_variable(shape):
    
    initial = tf.truncated_normal(shape, stddev = 0.1)
    
    return tf.Variable(initial)


'''
    Definiert die Bias-Terme in einer vollständig verbundenen oder in einer Konvolutionsschicht. 
    Werden mit 0,1 initialisiert.
'''
def bias_variable(shape):
    
    initial = tf.constant(0.1, shape = shape)
    
    return tf.Variable(initial)


'''
    Konvolusionsschicht, als vollständige Konvolution (ohne Überspringen von Bildpunkten),
    wodurch die Ausgabe so groß ist wie die Eingabe.
'''
def conv2d(x, W):
    
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

'''
    Max-Pooling auf die halbe Höhe und Breite der Eingabe, Output also ein Viertel vom Input
'''
def max_pool_2x2(x):
    
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

'''
    Die eigentliche Konvolutionsschicht, die mit conv2d definierte Konvolution mit einem Bias-Term,
    gefolgt von der nicht-linearen Aktivierungsfunktion ReLU.
'''
def conv_layer(input, shape):
    
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    
    return tf.nn.relu(conv2d(input, W) + b)

'''
    Vollverbundene Schicht mit Bias-Term. Wir haben hier keine ReLU-Aktivierungsfunktion hizugefügt.
    Damit können wir diese Funktion auch für die Ausgabeschicht verwenden, bei der wir den nicht-linearen
    Teil nicht benötigen.
'''
def full_layer(input, size):
    
    in_size = int(input.get_shape()[1])
    
    W = weight_variable([in_size, size])
    
    b = bias_variable([size])
    
    return tf.matmul(input, W) + b



''' ------------ CNN Model Generation -------------------- '''

''' 
    Zuerst nutzen wir dasselbe Model wie für die MNIST-Daten :-)
'''    

cifar = CifarDataManager()

x = tf.placeholder(tf.float32, shape = [None, 32, 32, 3]) # Achtung: MNIST-Bilder 28x28x1 Bildpunkte!
y_ = tf.placeholder(tf.float32, shape = [None, 10])

keep_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x, shape = [5, 5, 3, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape = [5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

conv3 = conv_layer(conv2_pool, shape = [5, 5, 64, 128])
conv3_pool = max_pool_2x2(conv3)
conv3_flat = tf.reshape(conv3_pool, [-1, 4 * 4 * 128])
conv3_drop = tf.nn.dropout(conv3_flat, keep_prob = keep_prob)

full_1 = tf.nn.relu(full_layer(conv3_drop, 512))
full1_drop = tf.nn.dropout(full_1, keep_prob = keep_prob)

y_conv = full_layer(full1_drop, 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def test(sess):
    
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    
    acc = np.mean([sess.run(accuracy, feed_dict = {x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(10)])
    
    
    
    print("Genauigkeit: {:.4}%".format(acc * 100))
    
with tf.Session() as sess:
    
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', graph = tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test', graph = tf.get_default_graph())
    
    sess.run(tf.global_variables_initializer())

    for i in range(NUM_STEPS):
        
        batch = cifar.train.next_batch(MINIBATCH_SIZE)
        
        if i % 10 == 0:
            
            ce_summary = tf.summary.scalar('Cross_Entropy', cross_entropy)
            #summary = ce_summary.eval(feed_dict = {x: batch[0], y_: batch[1]})
            
        
        sess.run(train_step, feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
        
        #train_writer.add_summary(summary, i)
        
        test(sess)


    train_writer.close()
    test_writer.close()
    
    
    
    
    




























