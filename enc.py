
dataset_file = '/home/mst/Desktop/HAC/Images/Radar/'

from tflearn.data_utils import image_preloader
X,Y = image_preloader(dataset_file, image_shape = (140,400),  mode = 'folder', categorical_labels = True,  normalize= True)
