
dataset_file = '/home/mst/Desktop/HAC/Images/Radar/'

from tflearn.data_utils import build_hdf5_image_dataset
build_hdf5_image_dataset(dataset_file, image_shape = None,  mode = 'folder', output_path = '/home/mst/Desktop/HAC/Images/dateset.h5', categorical_labels = True,  normalize= True)
