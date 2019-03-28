import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import numpy as np

X,Y, test_x, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1,28,28,1])
test_x = test_x.reshape([-1,28,28,1])

convnet = input_data(shape=[None,28,28,1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)


#model.fit({'input':X},{'targets':Y}, n_epoch=2, 
#	      validation_set=({'input':test_x},{'targets':test_y}), 
#	      snapshot_step=500, show_metric=True, run_id='mnist')

#saves weights
#model.save('tflearncnn.model')


model.load('tflearncnn.model')
print(np.round(model.predict([test_x[1]])[0]))
print(test_y[1])
#ili
print(model.predict([test_x[1]]))



