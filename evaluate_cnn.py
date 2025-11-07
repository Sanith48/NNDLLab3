from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


path = 'checkpoints/CNN-aug_best.h5'
outdir = 'plots'
os.makedirs(outdir, exist_ok=True)
print('Checkpoint exists:', os.path.exists(path))

if os.path.exists(path):
    m = load_model(path)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    y_test_cat = to_categorical(y_test, 10)

    # Evaluate
    loss, acc = m.evaluate(x_test, y_test_cat, verbose=2)
    print('CNN checkpoint test acc:', acc)

    # Predictions and report
    y_prob = m.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = y_test.reshape(-1)

    labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)

    # Save textual report
    report_path = os.path.join(outdir, 'CNN_eval_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'Checkpoint: {path}\n')
        f.write(f'Test loss: {loss:.6f}\n')
        f.write(f'Test acc: {acc:.6f}\n\n')
        f.write('Classification report:\n')
        f.write(report)
    print('Saved report to', report_path)

    # Save metrics JSON
    metrics = {'loss': float(loss), 'accuracy': float(acc)}
    with open(os.path.join(outdir, 'CNN_eval_metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('CNN Confusion Matrix')
    cm_path = os.path.join(outdir, 'CNN_confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print('Saved confusion matrix to', cm_path)

else:
    print('No checkpoint to evaluate')
