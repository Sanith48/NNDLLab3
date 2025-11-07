import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def build_cnn_model(input_shape=(32,32,3), num_classes=10):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def plot_history(history, name='cnn-aug', outdir='plots'):
    os.makedirs(outdir, exist_ok=True)
    h = history.history
    epochs = range(1, len(h.get('loss', [])) + 1)

    plt.figure(figsize=(8,4))
    plt.plot(epochs, h.get('loss', []), label='train_loss')
    plt.plot(epochs, h.get('val_loss', []), label='val_loss')
    plt.title(f'{name} loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f'{name}_loss.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(epochs, h.get('accuracy', h.get('acc', [])), label='train_acc')
    plt.plot(epochs, h.get('val_accuracy', h.get('val_acc', [])), label='val_acc')
    plt.title(f'{name} accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, f'{name}_acc.png'), bbox_inches='tight')
    plt.close()


def main():
    # Hyperparameters
    batch_size = 128
    epochs = 50
    lr = 1e-3

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    model = build_cnn_model()
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = os.path.join('checkpoints', 'CNN-aug_best.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    steps = max(1, len(x_train) // batch_size)
    history = model.fit(datagen.flow(x_train, y_train_cat, batch_size=batch_size),
                        steps_per_epoch=steps,
                        epochs=epochs,
                        validation_data=(x_test, y_test_cat),
                        callbacks=callbacks,
                        verbose=2)

    # Save final model
    final_path = os.path.join('checkpoints', 'CNN-aug_final.h5')
    model.save(final_path)
    print('Saved final model to', final_path)

    plot_history(history, name='CNN-aug', outdir='plots')
    print('Training complete. Best checkpoint saved to', ckpt_path)


if __name__ == '__main__':
    main()
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def build_cnn_model(input_shape=(32,32,3), num_classes=10):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def plot_history(history, out_dir='plots', name='cnn'):
    os.makedirs(out_dir, exist_ok=True)
    h = history.history
    plt.figure(figsize=(8,4))
    plt.plot(h.get('loss', []), label='train_loss')
    plt.plot(h.get('val_loss', []), label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(out_dir, f'{name}_loss.png'))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(h.get('accuracy', []), label='train_acc')
    plt.plot(h.get('val_accuracy', []), label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(out_dir, f'{name}_acc.png'))
    plt.close()


def main():
    # Hyperparameters
    batch_size = 128
    epochs = 50
    lr = 1e-3

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    model = build_cnn_model()
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    os.makedirs('checkpoints', exist_ok=True)
    ckpt = ModelCheckpoint('checkpoints/CNN-aug_best.h5', monitor='val_loss', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

    steps_per_epoch = max(1, len(x_train) // batch_size)
    history = model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(x_test, y_test_cat),
        callbacks=[ckpt, es, rlrop],
        verbose=2
    )

    # final evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
    print('Final test loss:', test_loss, 'test acc:', test_acc)

    plot_history(history, out_dir='plots', name='CNN-aug')


if __name__ == '__main__':
    main()
