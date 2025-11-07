import streamlit as st
import os
import io
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model


st.set_page_config(page_title='CIFAR-10 Demo', layout='centered')

# Prefer CNN checkpoint if present (better accuracy), fall back to feedforward checkpoint
cnn_path = os.path.join('checkpoints', 'CNN-aug_best.h5')
ffn_path = os.path.join('checkpoints', 'FFN-ReLU-full50_best.h5')
MODEL_PATH = cnn_path if os.path.exists(cnn_path) else ffn_path

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


@st.cache_data
def load_cifar_test():
    """Load CIFAR-10 test set and cache it to avoid re-downloading on reruns."""
    try:
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return x_test, y_test
    except Exception as e:
        # Return None to caller to handle
        return None, None


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

    # persist uploaded/selected image in session_state so it survives reruns
    if 'img_bytes' not in st.session_state:
        st.session_state.img_bytes = None
    if 'image_pil' not in st.session_state:
        st.session_state.image_pil = None

    if source == 'Upload image':
        uploaded = st.file_uploader('Upload an image (will be resized to 32x32)', type=['png', 'jpg', 'jpeg'], key='uploader')
        if uploaded is not None:
            try:
                b = uploaded.read()
                st.session_state.img_bytes = b
                st.session_state.image_pil = Image.open(BytesIO(b)).convert('RGB')
            except Exception as e:
                st.error(f'Failed to read uploaded image: {e}')
    else:
        if st.button('Load random CIFAR test image', key='load_random'):
            x_test, y_test = load_cifar_test()
            if x_test is None:
                st.error('Unable to load CIFAR-10 test set (no network or missing files).')
            else:
                idx = np.random.randint(0, len(x_test))
                arr = (x_test[idx]).astype('uint8')
                pil = Image.fromarray(arr)
                buf = BytesIO()
                pil.save(buf, format='PNG')
                st.session_state.img_bytes = buf.getvalue()
                st.session_state.image_pil = pil
                st.session_state.cifar_label = int(y_test[idx])
                st.success(f'Loaded CIFAR test image index {idx}')

    if st.session_state.image_pil is not None:
        st.image(st.session_state.image_pil, caption='Input image (resized to 32x32)', width=256)
        if 'cifar_label' in st.session_state:
            try:
                st.write(f"Label (ground truth): {LABELS[int(st.session_state.cifar_label)]}")
            except Exception:
                pass

        if model is None:
            st.info('Model missing — cannot run prediction.')
        else:
            st.success('Model loaded and ready')

            # optional sanity check to ensure model.predict works
            if st.button('Sanity check (model.predict random input)', key='sanity'):
                try:
                    sample = np.random.rand(1,32,32,3).astype('float32')
                    _p = model.predict(sample)
                    st.write('Sanity check OK — model produced output shape', _p.shape)
                except Exception as e:
                    st.error(f'Sanity check failed: {e}')

            arr = preprocess_pil(st.session_state.image_pil)
            if st.button('Predict', key='predict'):
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

                        import pandas as pd
                        df = pd.DataFrame({'label': top_labels, 'prob': top_probs})
                        st.bar_chart(df.set_index('label'))
                    except Exception as e:
                        st.error(f'Prediction failed: {e}')


if __name__ == '__main__':
    main()
