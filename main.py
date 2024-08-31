import os

from data_preprocessing import (create_df, data_generators, num_of_classes,
                                plot_class_distribution)
from evaluation import evaluate_model_performance, plot_training_results
from model import create_model
from training import train_model

# Define directories
base_dir = '/home/nico/fruit_vegetable_classifier/dataset/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Data preprocessing
classes = num_of_classes(train_dir, 'train')
train_df = create_df(train_dir, classes)
validation_df = create_df(validation_dir, classes)
test_df = create_df(test_dir, classes)
plot_class_distribution(train_dir)
train_generator, validation_generator, test_generator = data_generators(train_df, validation_df, test_df)

# Model creation and training
model = create_model()
history = train_model(model, train_generator, validation_generator)

# Evaluate model
evaluate_model_performance(model, validation_generator, classes)
plot_training_results(history)
