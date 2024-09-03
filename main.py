# Andrew Barber
# 09/02/2024
# Translator App, Python File

from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer
import json
import requests
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def load_model_and_tokenizer(target_language):
    model_name = {
        'es': 'Helsinki-NLP/opus-mt-en-es',
        'fr': 'Helsinki-NLP/opus-mt-en-fr',
        'de': 'Helsinki-NLP/opus-mt-en-de'
    }.get(target_language)
    
    if not model_name:
        return None, None

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text, target_language):
    model, tokenizer = load_model_and_tokenizer(target_language)
    if not model or not tokenizer:
        return "Error: Unsupported language"

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_text[0]

def extract_text_from_image(image_path):
    with open(image_path, "rb") as file:
        url = 'https://api.ocr.space/parse/image'
        files = {
            'filename': file
        }
        payload = {
            'isOverlayRequired': False,
            'apikey': "K83227554488957",
            'language': 'eng'
        }

        r = requests.post(url,data=payload,files=files)

    if r.status_code == 200:
        result = r.json()
        text = result.get("ParsedResults")[0].get("ParsedText")
        return text.strip()
    else:
        print(f"Error: {r.status_code}, {r.text}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ""
    if request.method == 'POST':
        text_to_translate = request.form['text']
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                text_to_translate = extract_text_from_image(file_path)
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        
        target_language = request.form['language']
        translated_text = translate_text(text_to_translate, target_language)
    return render_template('index.html', translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)