import os
import gradio as gr
from PIL import Image
import easyocr
import google.generativeai as genai

# OCR
reader = easyocr.Reader(['en'])

# Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def analyze(image):
    if image is None:
        return "Please upload a medicine image."

    # OCR
    result = reader.readtext(image)
    text = " ".join([r[1] for r in result])

    if not text:
        return "❌ No medicine name detected."

    prompt = f"""
You are a medical information assistant.

Medicine detected:
{text}

Give:
1. Medicine Name
2. Uses
3. Common Side Effects
4. General Precautions
5. Prescription Required? (Yes/No)

Do NOT prescribe treatment.
Do NOT give personalized medical advice.
Keep the answer short and easy to understand.
"""

    response = model.generate_content(prompt)

    return f"### OCR Detected\n{text}\n\n---\n\n{response.text}"


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="numpy", label="Upload Medicine"),
    outputs=gr.Markdown(),
    title="💊 AI Medicine Checker",
    description="Upload a medicine strip or bottle to identify it and get general information."
)

demo.launch()