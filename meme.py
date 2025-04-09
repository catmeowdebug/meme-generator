import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os
import torch
import textwrap
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load models with caching
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load BLIP for image captioning
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    # Load Hermes LLM for humor
    llm_name = "teknium/OpenHermes-2.5-Mistral-7B"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    ).to(device)

    return blip_processor, blip_model, llm_tokenizer, llm_model, device

blip_processor, blip_model, llm_tokenizer, llm_model, device = load_models()

# Get caption from image
def get_image_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# Make the caption funny using Hermes
def make_it_funny(caption):
    prompt = f"Make this caption funny for a meme: '{caption}'"
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = llm_model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.9)
    result = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split(":", 1)[-1].strip()

# Draw text on image
def create_meme(image, caption, font_path="anton.ttf"):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype(font_path, 36)
    except IOError:
        font = ImageFont.load_default()

    wrapped = textwrap.fill(caption, width=30)
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    text_width = bbox[2] - bbox[0]
    x = (image.width - text_width) / 2
    y = 10

    draw.text((x, y), wrapped, font=font, fill="white", stroke_width=2, stroke_fill="black")
    return image

# Streamlit UI
st.title("🧠 Meme Generator with AI")
st.write("Upload an image and let BLIP + Hermes LLM generate a funny meme!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    with st.spinner("Generating caption..."):
        caption = get_image_caption(image)

    st.markdown(f"**Caption:** {caption}")

    with st.spinner("Making it funny..."):
        funny_caption = make_it_funny(caption)

    st.markdown(f"**Funny Caption:** {funny_caption}")

    with st.spinner("Creating meme..."):
        meme = create_meme(image, funny_caption)
        st.image(meme, caption="Generated Meme")

        # Save and offer download
        meme.save("meme_output.jpg")
        with open("meme_output.jpg", "rb") as f:
            st.download_button("📥 Download Meme", f, file_name="funny_meme.jpg", mime="image/jpeg")
