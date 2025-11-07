from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

path='checkpoints/CNN-aug_best.h5'
print('Checkpoint exists:', os.path.exists(path))
if os.path.exists(path):
    m=load_model(path)
    (x_train,y_train),(x_test,y_test)=cifar10.load_data()
    x_test=x_test.astype('float32')/255.0
    y_test_cat=to_categorical(y_test,10)
    loss,acc=m.evaluate(x_test,y_test_cat,verbose=2)
    print('CNN checkpoint test acc:',acc)
else:
    print('No checkpoint to evaluate')
