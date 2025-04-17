import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import tempfile

# Load model and tokenizer once
caption_model = load_model("models/model.keras")
feature_extractor = load_model("models/feature_extractor.keras")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 34
img_size = 224


def generate_caption(image):
    # Save image temporarily
    temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    image.save(temp.name)

    # Preprocess
    img = load_img(temp.name, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(img, verbose=0)

    # Caption generation
    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = caption_model.predict([features, seq], verbose=0)
        word = tokenizer.index_word.get(np.argmax(yhat))
        if word is None or word == "endseq":
            break
        in_text += " " + word

    caption = in_text.replace("startseq", "").strip()
    return image, caption


# Gradio UI
with gr.Blocks(css="""
    #container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        max-width: 600px;
        margin: 0 auto;
    }
    .caption-text {
        font-size: 20px;
        font-weight: 500;
        text-align: center;
        color: #444;
        margin-top: 10px;
    }
""") as demo:
    gr.Markdown("<h1 style='text-align: center;'>🖼️ Image Caption Generator</h1>")
    with gr.Column(elem_id="container"):
        image_input = gr.Image(type="pil", label="Upload an Image")
        image_output = gr.Image(show_label=False, visible=False)
        caption_output = gr.Textbox(label="", elem_classes=["caption-text"], interactive=False)

    image_input.change(fn=generate_caption, inputs=image_input, outputs=[image_output, caption_output])

demo.launch()
