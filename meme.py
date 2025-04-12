import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Hugging Face API token
HF_TOKEN = "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"

# -----------------------------------------------
# Generate a caption from the uploaded image
# -----------------------------------------------
def generate_caption(image_bytes):
    url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/octet-stream"
    }
    response = requests.post(url, headers=headers, data=image_bytes)
    result = response.json()

    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif "error" in result:
        raise Exception(f"Captioning API Error: {result['error']}")
    else:
        raise Exception(f"Unexpected response: {result}")

# --------------------------------------------------
# Turn caption into a funny meme caption
# --------------------------------------------------
def generate_meme_line(caption):
    url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    prompt = (
        "### Instruction:\n"
        "Write a funny meme caption based on this image description.\n\n"
        f"### Image Description:\n{caption}\n\n"
        "### Meme Caption:"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.8,
            "max_new_tokens": 60,
            "do_sample": True,
            "top_p": 0.95
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    if isinstance(result, list):
        text = result[0].get("generated_text", "")
    elif "generated_text" in result:
        text = result["generated_text"]
    elif "error" in result:
        raise Exception(f"Meme Generator Error: {result['error']}")
    else:
        raise Exception("Unknown error from meme generation.")

    if "### Meme Caption:" in text:
        return text.split("### Meme Caption:")[-1].strip()
    return text.strip()

# -----------------------------------------------------------------
# Add meme caption below the image (extend canvas with text)
# -----------------------------------------------------------------
def add_caption_below_image(image, caption_text):
    width, height = image.size

    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    draw_temp = ImageDraw.Draw(Image.new("RGB", (width, 1)))
    words = caption_text.split()
    lines = []
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        if draw_temp.textlength(test_line, font=font) <= width - 40:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)

    line_height = font.getbbox("A")[3] - font.getbbox("A")[1] + 10
    caption_height = line_height * len(lines) + 20

    new_image = Image.new("RGB", (width, height + caption_height), color="white")
    new_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_image)

    y = height + 10
    for line in lines:
        text_width = draw.textlength(line, font=font)
        x = (width - text_width) / 2
        draw.text((x, y), line, font=font, fill="black")
        y += line_height

    return new_image

# ---------------------------------------------------
# 🌐 Streamlit Interface
# ---------------------------------------------------
st.title("😂 AI Meme Generator")
st.caption("Upload an image and let the AI caption and meme it!")

uploaded_image = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner("Generating meme..."):
        try:
            # Step 1: Convert image to bytes
            with BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                image_bytes = buffer.getvalue()

            # Step 2: Generate caption and meme
            caption = generate_caption(image_bytes)
            meme_text = generate_meme_line(caption)

            # Step 3: Add meme to image
            meme_image = add_caption_below_image(image, meme_text)

            st.success("Meme generated successfully!")
            st.image(meme_image, caption="Meme Image", use_column_width=True)
            st.text_area("Meme Text", meme_text, height=100)

            # Optional: Download
            img_buffer = BytesIO()
            meme_image.save(img_buffer, format="JPEG")
            st.download_button("Download Meme", data=img_buffer.getvalue(), file_name="meme.jpg", mime="image/jpeg")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
