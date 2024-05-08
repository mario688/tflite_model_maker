# Imports
import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Confirm TF Version
print("\nTensorflow Version:")
print(tf.__version__)
print()


# LOAD DATASET 
train_data = object_detector.DataLoader.from_pascal_voc(
    'train-dataset/train', # <- match folder name 
    'train-dataset/train', # <- match folder name 
    ['Target'] # <- set labels  
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'train-dataset/valid', # <- match folder name 
    'train-dataset/valid', # <- match folder name 
    ['Target'] # <- set labels  
)
# CHOOSE MODEL 0,1,2,3,4
# info about models: https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector/EfficientDetLite1Spec?hl=en
spec = object_detector.EfficientDetLite2Spec()

# SET batch_size & epochs
model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=6, validation_data=val_data)
# Evaluate the model
eval_result = model.evaluate(val_data)

# Print COCO metrics
print("COCO metrics:")
for label, metric_value in eval_result.items():
    print(f"{label}: {metric_value}")

# Add a line break after all the items have been printed
print()

# Export the model
model.export(export_dir='.', tflite_filename='target.tflite')

# Evaluate the tflite model
tflite_eval_result = model.evaluate_tflite('target.tflite', val_data)

# Print COCO metrics for tflite
print("COCO metrics tflite")
for label, metric_value in tflite_eval_result.items():
    print(f"{label}: {metric_value}")
