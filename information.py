#step1 : setup api key of groq
import os
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

#step2 : convert image to required format
#step3 : setup multimodal llm

# information.py
import base64
from groq import Groq


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_query(query, model, encoded_image):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]
    response = client.chat.completions.create(
        messages=messages,
        model=model
    )
    return response.choices[0].message.content
