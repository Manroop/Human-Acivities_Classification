
import tflearn
from tflearn.data_utils import image_preloader
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


# Data Preprocessing
training_files = '/home/mst/Documents/CNNN/Train'
test_files = '/home/mst/Documents/CNNN/Test'
validation_files = '/home/mst/Documents/CNNN/Validation'


X,Y = image_preloader(training_files, image_shape = (140,220),  mode = 'folder', categorical_labels = True,  normalize= True)
test_X, test_Y = image_preloader(test_files, image_shape = (140,220),  mode = 'folder', categorical_labels = True,  normalize= True)
valid_X, valid_Y = image_preloader(validation_files, image_shape = (140,220),  mode = 'folder', categorical_labels = True,  normalize= True)


# Building convolutional network
network = input_data(shape=[None, 220, 140,3], name='input')
network = conv_2d(network, 5, 5, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 5, 5, activation='relu', regularizer="L2")
network = max_pool_2d(network, 4)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
#network = fully_connected(network, 256, activation='tanh')
#network = dropout(network, 0.8)
network = fully_connected(network, 7, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,batch_size = 24,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=3, tensorboard_dir='/home/mst/Documents/CNNN/board_files/')
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': valid_X}, {'target': valid_Y}),
snapshot_step=100, show_metric=True, run_id='Training_Phase_14')



model.save('my_model.tflearn')
