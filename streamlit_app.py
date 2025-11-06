import streamlit as st
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


st.set_page_config(page_title='CIFAR-10 Demo', layout='centered')

MODEL_PATH = os.path.join('checkpoints', 'FFN-ReLU-full50_best.h5')

LABELS = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


@st.cache_resource
def load_trained_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        model = load_model(path)
        return model
    except Exception as e:
        # Loading can fail silently inside Streamlit; surface the error
        st.error(f"Failed to load model from {path}: {e}")
        return None


def preprocess_pil(img: Image.Image):
    # Ensure RGB and resize to 32x32
    img = img.convert('RGB').resize((32, 32))
    arr = np.array(img).astype('float32') / 255.0
    # Model expects shape (1,32,32,3)
    return np.expand_dims(arr, axis=0)


def predict_and_display(model, img_array):
    preds = model.predict(img_array)
    probs = preds.flatten()
    top_idx = np.argsort(probs)[::-1][:5]
    top_probs = probs[top_idx]
    top_labels = [LABELS[i] for i in top_idx]

    st.write('Top predictions:')
    for label, p in zip(top_labels, top_probs):
        st.write(f"- {label}: {p*100:.2f}%")


def main():
    st.title('CIFAR-10 — Demo (Feedforward model)')
    st.write('This demo loads a trained Keras model from `checkpoints/FFN-ReLU-full50_best.h5` and predicts the class of an input image (32x32, RGB).')

    model = load_trained_model()
    if model is None:
        st.warning('Trained model not found in `checkpoints/FFN-ReLU-full50_best.h5`. Run the notebook to train and create the checkpoint, or place a compatible Keras .h5 model at that path.')

    source = st.radio('Image source', ['Upload image', 'Random CIFAR-10 test image'])

    image_to_show = None
    if source == 'Upload image':
        uploaded = st.file_uploader('Upload an image (will be resized to 32x32)', type=['png', 'jpg', 'jpeg'])
        if uploaded is not None:
            img = Image.open(uploaded)
            image_to_show = img
    else:
        # load CIFAR test images and pick a random one
        if st.button('Load random CIFAR test image'):
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            idx = np.random.randint(0, len(x_test))
            arr = (x_test[idx]).astype('uint8')
            image_to_show = Image.fromarray(arr)
            st.write(f'Label (ground truth): {LABELS[int(y_test[idx])] if y_test is not None else "-"}')

    if image_to_show is not None:
        st.image(image_to_show, caption='Input image (resized to 32x32)', use_container_width=False)

        if model is None:
            st.info('Model missing — cannot run prediction.')
        else:
            st.success('Model loaded and ready')
            arr = preprocess_pil(image_to_show)
            if st.button('Predict'):
                with st.spinner('Running prediction...'):
                    try:
                        preds = model.predict(arr)
                        probs = preds.flatten()
                        top_idx = np.argsort(probs)[::-1][:5]
                        top_probs = probs[top_idx]
                        top_labels = [LABELS[i] for i in top_idx]

                        st.write('Top predictions:')
                        for label, p in zip(top_labels, top_probs):
                            st.write(f"- {label}: {p*100:.2f}%")

                        # show a simple bar chart of top-5
                        import pandas as pd
                        df = pd.DataFrame({'label': top_labels, 'prob': top_probs})
                        st.bar_chart(df.set_index('label'))
                    except Exception as e:
                        st.error(f'Prediction failed: {e}')


if __name__ == '__main__':
    main()
