import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "image-classification",
    model="watersplash/waste-classification",
)

LABELS = {
    "battery": "🔋 Battery",
    "biological": "🥦 Biological / Food Waste",
    "brown-glass": "🟤 Brown Glass",
    "cardboard": "📦 Cardboard",
    "clothes": "👕 Clothes / Textile",
    "green-glass": "🟢 Green Glass",
    "metal": "🔩 Metal",
    "paper": "📄 Paper",
    "plastic": "🧴 Plastic",
    "shoes": "👟 Shoes",
    "trash": "🗑️ Miscellaneous Trash",
    "white-glass": "⬜ White Glass",
}


def classify(image):
    if image is None:
        return {}
    results = classifier(image, top_k=5)
    return {
        LABELS.get(r["label"], r["label"]): round(r["score"], 4)
        for r in results
    }


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a waste image"),
    outputs=gr.Label(num_top_classes=5, label="Waste Category"),
    title="Waste Image Classifier",
    description=(
        "Upload a photo of waste and the model will classify it into one of 12 categories. "
        "Built with a Vision Transformer (ViT) fine-tuned on waste images."
    ),
    examples=[
        ["data/RealWaste/1-Cardboard/Cardboard_1.jpg"],
        ["data/RealWaste/3-Glass/Glass_1.jpg"],
        ["data/RealWaste/4-Metal/Metal_1.jpg"],
        ["data/RealWaste/7-Plastic/Plastic_1.jpg"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
