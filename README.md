😂 AI Meme Generator
This web app lets you upload an image and generates a hilarious meme caption using two powerful AI models:

BLIP for image captioning

Zephyr for generating funny meme text

It automatically adds the caption below the image (so it doesn’t block the visual), and lets you download the final meme.

🖼 Demo

“When AI gets too funny for its own good…”

🚀 Features
Upload any image (cats encouraged 🐱).

AI generates a description of the image.

A second AI model writes a meme caption based on the description.

The meme caption is neatly added below the image.

Download your masterpiece in one click!

🛠 Technologies
Streamlit – UI and interactivity

Pillow (PIL) – Image handling and annotation

Hugging Face Inference API

Salesforce/blip-image-captioning-base

HuggingFaceH4/zephyr-7b-alpha

📦 Setup Instructions
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/your-username/ai-meme-generator.git
cd ai-meme-generator
2. Create a Hugging Face token
Go to huggingface.co

Create a Read access token

Replace the value of HF_TOKEN in meme.py

3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the app
bash
Copy
Edit
streamlit run app.py
🧠 Example Use Case
Upload Image	Generated Meme
"When you realize the Zoom call wasn't muted..."
📄 requirements.txt
txt
Copy
Edit
streamlit
requests
Pillow
☁️ Deploy to Streamlit Cloud
Push this project to GitHub.

Go to streamlit.io/cloud and link your repo.

Done. 🎉 Share the meme-making magic!

📝 License
This project is open-source under the MIT License.

🤓 Author
Made with ❤️ by YourName
Powered by memes and caffeine ☕💬

Let me know if you'd like a badge version, or want me to add preview images or a LICENSE file too!








