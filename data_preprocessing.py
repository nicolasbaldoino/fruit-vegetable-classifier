import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from termcolor import colored


def num_of_classes(folder_dir, folder_name):
    classes = [class_name for class_name in os.listdir(folder_dir)]
    print(colored(f'Number of classes in {folder_name} folder: {len(classes)}', 'blue', attrs=['bold']))
    return classes

def create_df(folder_path, classes):
    all_images = []
    for class_name in classes:
        class_path = os.path.join(folder_path, class_name)
        all_images.extend([(os.path.join(class_path, file_name), class_name) for file_name in os.listdir(class_path)])
    df = pd.DataFrame(all_images, columns=['file_path', 'label'])
    return df

def plot_class_distribution(train_dir):
    classes = [class_name for class_name in os.listdir(train_dir)]
    count = [len(os.listdir(os.path.join(train_dir, class_name))) for class_name in classes]
    
    plt.figure(figsize=(15, 4))
    ax = sns.barplot(x=classes, y=count, color='navy')
    plt.xticks(rotation=285)
    for i in ax.containers:
        ax.bar_label(i,)
    plt.title('Number of samples per label', fontsize=25, fontweight='bold')
    plt.xlabel('Labels', fontsize=15)
    plt.ylabel('Counts', fontsize=15)
    plt.yticks(np.arange(0, 105, 10))
    plt.show()

def data_generators(train_df, validation_df, test_df):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='file_path',
        y_col='label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validation_df,
        x_col='file_path',
        y_col='label',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=32,
        seed=42,
        shuffle=False
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='file_path',
        y_col='label',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=32,
        seed=42,
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator
