import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import models


def evaluate_model_performance(model, val_generator, class_labels):
    """
    Evaluate the model's performance on the validation set and print the classification report.
    """
    true_labels = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())
    
    predictions = model.predict(val_generator, steps=len(val_generator))
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Classification report
    report = classification_report(true_labels, predicted_labels, target_names=class_labels)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(15,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_training_results(history):
    result_df = pd.DataFrame(history.history)
    x = np.arange(len(result_df))
    
    fig, ax = plt.subplots(3, 1, figsize=(15, 12))
    ax[0].plot(x, result_df.loss, label='loss', linewidth=3)
    ax[0].plot(x, result_df.val_loss, label='val_loss', linewidth=2, ls='-.', c='r')
    ax[0].set_title('Loss', fontsize=20)
    ax[0].legend()
    
    ax[1].plot(x, result_df.accuracy, label='accuracy', linewidth=2)
    ax[1].plot(x, result_df.val_accuracy, label='val_accuracy', linewidth=2, ls='-.', c='r')
    ax[1].set_title('Accuracy', fontsize=20)
    ax[1].legend()
    
    ax[2].plot(x, result_df.lr, label='learning_rate', linewidth=2)
    ax[2].set_title('Learning Rate', fontsize=20)
    ax[2].set_xlabel('Epochs')
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()
