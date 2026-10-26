import os, json, numpy as np, tensorflow as tf, gradio as gr
from tensorflow import keras

IMG = 224
model = keras.models.load_model("oral_model.keras")
class_names = json.load(open("class_names.json"))

INFO = {
    "Calculus": "Hardened plaque (tartar) on the teeth, usually near the gum line.",
    "Gingivitis": "Inflamed, red or swollen gums, often from plaque build-up.",
    "Hypodontia": "One or more teeth are missing because they never developed.",
    "Mouth ulcer": "A painful sore or open lesion inside the mouth.",
    "Tooth discoloration": "Teeth that are yellow, brown or stained.",
}

def predict(image):
    if image is None:
        return {}, "### Please upload an image first.", ""
    img = tf.image.resize(np.array(image.convert("RGB")), (IMG, IMG))
    probs = model.predict(tf.expand_dims(img, 0), verbose=0)[0]
    result = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    top, conf = class_names[int(probs.argmax())], float(probs.max())
    if conf < 0.6:
        verdict = (f"### ⚠️ Low confidence ({conf:.0%})\n"
                   "The image may be unclear or may not be a mouth/teeth photo. Try a clearer, closer photo.")
        about = ""
    else:
        verdict = f"### ✅ Most likely: {top}\nConfidence: **{conf:.0%}**"
        about = f"**About {top}:** {INFO.get(top, '')}"
    return result, verdict, about

CSS = """
.gradio-container {max-width: 1000px !important; margin: auto;}
#title {text-align: center;}
#predict-btn {font-size: 18px;}
footer {display: none !important;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal"), css=CSS, title="Oral Diseases Classifier") as demo:
    gr.Markdown("# 🦷 Oral Diseases Classifier\n"
                "Upload a clear photo of the mouth or teeth and the AI will predict the condition.",
                elem_id="title")

    with gr.Row(equal_height=True):
        with gr.Column():
            img = gr.Image(type="pil", label="Upload an oral image", height=360)
            with gr.Row():
                clear_btn = gr.ClearButton(value="🗑️ Clear")
                submit_btn = gr.Button("🔍 Predict", variant="primary", elem_id="predict-btn")
        with gr.Column():
            verdict = gr.Markdown("### Result will appear here")
            label = gr.Label(num_top_classes=5, label="Class probabilities")
            about = gr.Markdown()

    if os.path.isdir("examples"):
        files = [os.path.join("examples", f) for f in sorted(os.listdir("examples"))
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        if files:
            gr.Examples(examples=files, inputs=img, label="Try an example")

    with gr.Accordion("ℹ️ About this model", open=False):
        gr.Markdown("- Model: **MobileNetV2** (transfer learning), 5 classes.\n"
                    "- Test accuracy: **95.6%**, macro-F1: **0.93**.\n"
                    "- The model only knows 5 conditions, so it will always pick one of them, even for a non-teeth image.")

    gr.Markdown("*For educational purposes only. This is not a medical diagnosis. Please consult a dentist.*")

    clear_btn.add([img, label, verdict, about])
    submit_btn.click(predict, inputs=img, outputs=[label, verdict, about])
    img.upload(predict, inputs=img, outputs=[label, verdict, about])

if __name__ == "__main__":
    demo.launch()