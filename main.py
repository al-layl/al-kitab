import os
import io
import time
import re
import openai
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision
import arabic_reshaper
from bidi.algorithm import get_display
from dotenv import load_dotenv

# === Setup ===
BASE_DIR = os.path.dirname(__file__)
load_dotenv()
openai.api_key = os.getenv("OPENAIKEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(BASE_DIR, "static-mile-460504-q5-253eccbdfafa.json")

# === Translate with OpenL ===
def translate_openl(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You're an expert Arabic manga translator. Translate English text into spoken Palestinian Shami Arabic with natural, emotionally expressive phrasing like you'd hear in a conversation or dubbed anime.\n\n"
                        "✅ Follow these rules:\n"
                        "- Use only Palestinian Levantine (no Modern Standard Arabic).\n"
                        "- Avoid literal structure—translate ideas, not grammar.\n"
                        "- Keep emotional tone, slang, and phrasing local to Palestinian speech.\n"
                        "- Use connectors like: إنو، هاد، هيك، شو، ليش، بقلك، مهاجمينا، هالحكي، بكون، بدنا، الخ…\n"
                        "- Masculine/feminine forms must match.\n"
                        "- Preserve the original artist’s intent and emotional weight of the scene.\n"
                        "- Add punctuation that reflects tone (!؟…) but **don’t add or invent** content.\n\n"
                        "❌ Wrong: 'من كان يخبر الجميع أنهم يهاجموننا؟'\n"
                        "✅ Right: 'مين قال للكل إنو كانوا مهاجمينا؟'\n\n"
                        "Only return the Arabic translation."
                    )
                },
                {"role": "user", "content": text}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error translating '{text}': {e}")
        return text

# === Google OCR ===
def google_ocr(image_path):
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(f"Google OCR error: {response.error.message}")

    sentence_boxes = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                sentence_text = ""
                x_coords, y_coords = [], []

                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    sentence_text += word_text + " "
                    for vertex in word.bounding_box.vertices:
                        x_coords.append(vertex.x)
                        y_coords.append(vertex.y)

                clean_text = sentence_text.strip()
                area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                if len(clean_text) <= 6 and area > 40000:
                    continue
                if not re.search(r'\w', clean_text):
                    continue
                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                sentence_boxes.append({"text": clean_text, "bbox": bbox})

    return sentence_boxes

# === Wrap Arabic Text ===
def wrap_text(text, draw, font, max_width):
    words = text.split()
    lines = []
    current_line = ''
    for word in words:
        test_line = word if not current_line else f"{current_line} {word}"
        width = draw.textbbox((0, 0), test_line, font=font)[2]
        if width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

# === Erase & Draw Text ===
def erase_sentences_from_image(image_path, sentence_boxes, output_path):
    invalid_phrases = [
        "please provide the text", "cannot translate the text",
        "doesn’t convey a clear meaning", "can’t assist with that",
        "does not appear to be a coherent phrase",
        "I'm sorry, but I can't assist with that.",
    ]

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font_path = os.path.join(BASE_DIR, "Noto_Naskh_Arabic", "NotoNaskhArabic-VariableFont_wght.ttf")
    padding_x, padding_y = 12, 10

    for box in sentence_boxes:
        x0, y0, x1, y1 = box["bbox"]
        x0 = max(0, x0 - padding_x)
        y0 = max(0, y0 - padding_y)
        x1 += padding_x
        y1 += padding_y
        draw.rectangle([(x0, y0), (x1, y1)], fill="white")

    for box in sentence_boxes:
        text = box["text"].strip()
        if not text or text.isspace():
            continue

        x0, y0, x1, y1 = box["bbox"]
        x0 = max(0, x0 - padding_x)
        y0 = max(0, y0 - padding_y)
        x1 += padding_x
        y1 += padding_y
        max_width = x1 - x0
        max_height = y1 - y0
        if max_width < 10 or max_height < 10:
            continue

        translated_text = translate_openl(text)
        if any(p in translated_text.lower() for p in invalid_phrases):
            print(f"Skipping invalid translation: {translated_text}")
            continue
        translated_text = re.sub(r'[A-Za-z]', '', translated_text).strip()
        if not translated_text or translated_text.isspace():
            print("Translation was empty after removing English.")
            continue

        font_size = 23
        min_font_size = 18
        chosen_font = None
        wrapped_lines = []
        reshaped_lines = []

        while font_size >= min_font_size:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = ImageFont.load_default()

            wrapped_lines = wrap_text(translated_text, draw, font, max_width)
            reshaped_lines = [get_display(arabic_reshaper.reshape(line)) for line in wrapped_lines]

            line_height = draw.textbbox((0, 0), "Test", font=font)[3]
            total_height = line_height * len(reshaped_lines) + (len(reshaped_lines) - 1) * 4

            if total_height <= max_height:
                chosen_font = font
                break
            font_size -= 2

        if not chosen_font:
            font = ImageFont.truetype(font_path, min_font_size)
            wrapped_lines = wrap_text(translated_text, draw, font, max_width)
            reshaped_lines = [get_display(arabic_reshaper.reshape(line)) for line in wrapped_lines]
            line_height = draw.textbbox((0, 0), "Test", font=font)[3]
            total_height = line_height * len(reshaped_lines) + (len(reshaped_lines) - 1) * 4
        else:
            font = chosen_font

        current_y = y0 + ((max_height - total_height) // 2)
        for line in reshaped_lines:
            line_width = draw.textbbox((0, 0), line, font=font)[2]
            center_x = x0 + ((max_width - line_width) // 2)
            draw.text((center_x, current_y), line, fill="black", font=font)
            current_y += line_height + 4

    image.save(output_path)
    print(f"✅ Translated and saved: {output_path}")

# === Batch Processor ===
def process_folder(folder_path, output_folder):
    supported_exts = [".jpg", ".jpeg", ".png", ".webp"]
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in supported_exts):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, f"translated_{filename}")
            print(f"🔄 Processing: {filename}")
            sentences = google_ocr(input_path)
            erase_sentences_from_image(input_path, sentences, output_path)

# === Run ===
input_folder = os.path.join(BASE_DIR, "images")
output_folder = os.path.join(BASE_DIR, "output")
process_folder(input_folder, output_folder)
