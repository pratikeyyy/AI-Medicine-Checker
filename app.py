import os
import json
import re
import tempfile
import gradio as gr
from PIL import Image
import easyocr
import google.generativeai as genai
from fpdf import FPDF
from gtts import gTTS
import numpy as np

# --- Initialization ---
reader = easyocr.Reader(['en'], gpu=False)

if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
else:
    # Fallback to prevent immediate crash if key is missing during UI build
    genai.configure(api_key="DUMMY_KEY") 

# --- Custom CSS ---
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
    --bg-color: #0B1220;
    --card-bg: #111827;
    --primary: #14B8A6;
    --accent: #3B82F6;
    --success: #22C55E;
    --warning: #F59E0B;
    --danger: #EF4444;
    --text-main: #F9FAFB;
    --text-muted: #9CA3AF;
    --border-color: #374151;
}
.stat-card{
background:#172554;
padding:20px;
border-radius:18px;
width:220px;
border:1px solid #334155;
transition:.3s;
}
.stat-card:hover{
transform:translateY(-8px);
box-shadow:0 15px 35px rgba(0,0,0,.35);
}
.stat-card h2{
margin:0;
font-size:20px;
color:white;
}
.stat-card p{
color:#CBD5E1;
margin-top:10px;
}
body {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: linear-gradient(135deg,#0B1220,#111827,#0F172A) !important;
    color: var(--text-main) !important;
}
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    background: transparent !important;
    border: none !important;
}
.hero-section {
    text-align: center;
    padding: 3rem 1rem 4rem;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    color: var(--text-main);
    margin-bottom: 1.25rem;
    letter-spacing: -0.05em;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.25rem;
    color:#CBD5E1;
    max-width: 600px;
    margin: 0 auto;
    font-weight: 400;
    line-height: 1.6;
}
.result-container {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}
.med-card {
    background: var(--card-bg);
    border-radius: 20px;
    padding: 1.75rem;
    box-shadow: 0 20px 40px rgba(0,0,0,.45);
    border: 1px solid var(--border-color);
    border-left: 6px solid var(--primary);
}
.med-card.success { border-left-color: var(--success); }
.med-card.warning { border-left-color: var(--warning); }
.med-card.danger { border-left-color: var(--danger); }
.med-card.accent { border-left-color: var(--accent); }
.med-card h3 {
    margin-top: 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: var(--text-main);
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 1.25rem;
}
.grid-2 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.25rem;
}
.data-label {
    font-size: 0.875rem;
    color: var(--text-muted);
    font-weight: 500;
    margin-bottom: 0.25rem;
    display: block;
}
.data-value {
    font-size: 1rem;
    color: var(--text-main);
    font-weight: 600;
}
.highlight {
    color: var(--primary);
    font-size: 1.25rem;
    font-weight: 700;
}
ul.styled-list {
    list-style-type: none;
    padding: 0;
    margin: 0;
}
ul.styled-list li {
    position: relative;
    padding-left: 1.5rem;
    margin-bottom: 0.5rem;
    color: var(--text-main);
    line-height: 1.5;
}
ul.styled-list li::before {
    content: "•";
    position: absolute;
    left: 0;
    color: var(--primary);
    font-weight: bold;
    font-size: 1.2rem;
}
.footer {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-muted);
    font-size: 0.875rem;
    border-top: 1px solid var(--border-color);
    margin-top: 5rem;
}
"""

# --- Helper Functions ---
def clean_json_response(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def generate_report_pdf(data):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Safely encode text to prevent unicode runtime errors with FPDF
        def safe_text(text):
            return str(text).encode('latin-1', 'ignore').decode('latin-1')
            
        pdf.cell(200, 10, txt="MediVision AI Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Medication Name: {safe_text(data.get('name', 'Unknown'))}", ln=1, align='L')
        pdf.cell(200, 10, txt=f"Purpose: {safe_text(data.get('purpose', 'Unknown'))}", ln=1, align='L')
        pdf.cell(200, 10, txt=f"Dosage: {safe_text(data.get('dosage', 'Not specified'))}", ln=1, align='L')
        pdf.cell(200, 10, txt="Warnings:", ln=1, align='L')
        
        for w in data.get("warnings", ["None"]):
            pdf.cell(200, 10, txt=f"- {safe_text(w)}", ln=1, align='L')
            
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_file.name)
        return temp_file.name
    except Exception:
        return None

def generate_voice_summary(data):
    try:
        text = f"Medication Name: {data.get('name', 'Unknown')}. Purpose: {data.get('purpose', 'Unknown')}."
        tts = gTTS(text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception:
        return None

def get_best_gemini_model():
    """Dynamically finds the correct model name regardless of SDK version"""
    chosen_model = None
    
    # Find all models that support generating text
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            # Prefer flash if it exists (faster, cheaper)
            if 'flash' in m.name:
                return m.name
            # Keep track of any working model as a backup
            chosen_model = m.name
            
    if not chosen_model:
        raise Exception("No text generation models found for this API key.")
        
    return chosen_model

def analyze(img):
    if "GEMINI_API_KEY" not in os.environ or os.environ["GEMINI_API_KEY"] == "DUMMY_KEY":
        return """
        <div class="result-container">
            <div class="med-card warning">
                <h3>⚠️ Missing API Key</h3>
                <p>Please configure your <b>GEMINI_API_KEY</b> in the environment variables.</p>
            </div>
        </div>
        """, None, None

    if img is None:
        return """
        <div class="result-container">
            <div class="med-card warning">
                <h3>⚠️ No Image</h3>
                <p>Please upload an image before analyzing.</p>
            </div>
        </div>
        """, None, None

    try:
        # 1. OCR text extraction
        ocr_results = reader.readtext(img)
        extracted_text = " ".join([res[1] for res in ocr_results])

        if not extracted_text.strip():
            return """
            <div class="result-container">
                <div class="med-card danger">
                    <h3>❌ Analysis Failed</h3>
                    <p>Could not read any text from the image. Please try a clearer picture.</p>
                </div>
            </div>
            """, None, None

        # 2. Bulletproof AI Connection
        model_name = get_best_gemini_model()
        model = genai.GenerativeModel(model_name)

        prompt = f'''
        You are a medical AI assistant. Extract and infer the following details from this raw OCR text taken from a medicine package:
        "{extracted_text}"
        Return ONLY a JSON object with the following keys:
        - "name": Medication name (String)
        - "purpose": Primary use/indication (String)
        - "warnings": A list of side effects or precautions (List of Strings)
        - "dosage": Suggested dosage or instructions if visible (String)
        '''

        # 3. AI Generation
        response = model.generate_content(prompt)
        
        cleaned_json = clean_json_response(response.text)
        data = json.loads(cleaned_json)

        # 4. Format Output
        warnings_html = "".join([f"<li>{w}</li>" for w in data.get("warnings", ["No specific warnings detected."])])

        html_output = f"""
        <div class="result-container">
            <div class="med-card success">
                <h3>✅ Analysis Complete</h3>
                <div class="grid-2">
                    <div>
                        <span class="data-label">Medication Name</span>
                        <span class="data-value highlight">{data.get('name', 'Unknown')}</span>
                    </div>
                    <div>
                        <span class="data-label">Purpose</span>
                        <span class="data-value">{data.get('purpose', 'Unknown')}</span>
                    </div>
                    <div>
                        <span class="data-label">Dosage Instructions</span>
                        <span class="data-value">{data.get('dosage', 'Not specified')}</span>
                    </div>
                </div>
            </div>
            <div class="med-card danger">
                <h3>⚠️ Warnings & Precautions</h3>
                <ul class="styled-list">
                    {warnings_html}
                </ul>
            </div>
        </div>
        """
        
        pdf_path = generate_report_pdf(data)
        audio_path = generate_voice_summary(data)
        
        return html_output, pdf_path, audio_path

    except Exception as e:
        return f"""
        <div class="result-container">
            <div class="med-card danger">
                <h3>❌ System Error</h3>
                <p>An error occurred: {str(e)}</p>
            </div>
        </div>
        """, None, None

# --- Gradio UI ---
with gr.Blocks(css=css, title="MediVision AI") as demo:
    gr.HTML("""
    <div class="hero-section">
        <h1 class="hero-title">💊 MediVision AI</h1>
        <p class="hero-subtitle">
            AI Powered Medicine Recognition & Safety Assistant
        </p>
        <div style="display:flex;justify-content:center;gap:18px;flex-wrap:wrap;margin-top:30px;">
            <div class="stat-card">
                <h2>⚡ Fast OCR</h2>
                <p>Instant medicine detection</p>
            </div>
            <div class="stat-card">
                <h2>🧠 Gemini AI</h2>
                <p>Smart medicine explanation</p>
            </div>
            <div class="stat-card">
                <h2>📄 PDF Report</h2>
                <p>Download complete report</p>
            </div>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(
                type="numpy",
                label="📷 Upload Medicine",
                sources=["upload", "webcam"],
                height=450
            )

            analyze_btn = gr.Button(
                "🚀 Scan Medicine",
                variant="primary"
            )

            clear_btn = gr.ClearButton(components=[image])

        with gr.Column(scale=1):
            output = gr.HTML("""
            <div class="result-container">
                <div class="med-card accent">
                    <h3>🩺 Ready</h3>
                    <p>Upload a medicine image to begin analysis.</p>
                </div>
            </div>
            """)
                
            with gr.Row():
                pdf_out = gr.File(label="Download Report")
                audio_out = gr.Audio(label="Listen to Summary")

    analyze_btn.click(
        fn=analyze,
        inputs=image,
        outputs=[output, pdf_out, audio_out]
    )

    gr.HTML("""
    <div class="footer">
        ⚠️ Educational purposes only. Not professional medical advice.
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
