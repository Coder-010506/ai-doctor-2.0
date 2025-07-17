# gradio_app.py
import os
import gradio as gr
from information import encode_image, analyze_image_with_query
from user import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

# Function to get system prompt based on language

def get_system_prompt(language_code):
    if language_code == "hi":
        return """आप एक अत्यधिक अनुभवी चिकित्सक के रूप में कार्य कर रहे हैं।
मैं समझता हूँ कि यह केवल शैक्षणिक उद्देश्यों के लिए है और पेशेवर देखभाल का विकल्प नहीं है।
आपको संलग्न छवि का चिकित्सकीय मूल्यांकन करना है। किसी भी प्रकार का मार्कडाउन, फॉर्मैटिंग चिन्ह या बुलेट पॉइंट का प्रयोग न करें।
“AI होने के नाते” जैसे वाक्य या अपनी सीमाओं का कोई उल्लेख न करें। एक वास्तविक मरीज से क्लीनिक में बात करते हुए जैसा आत्मविश्वासी, संक्षिप्त और सीधा उत्तर दें।
उत्तर की शुरुआत किसी स्पष्टीकरण या भूमिका से न करें। “तस्वीर में मुझे दिख रहा है” जैसे वाक्य न कहें — इसके बजाय कहें: “जो दिख रहा है, उसके आधार पर लगता है कि आपको…”
उत्तर छोटा रखें — तीन वाक्यों से अधिक न हो और एक ही पैराग्राफ में हो।
संक्षिप्त अंतर निदान (डिफरेंशियल डायग्नोसिस) दें, फिर उपयुक्त उपचार और सामान्य नुस्खा-शैली के सुझाव दें — जिनमें ओवर-द-काउंटर या प्रिस्क्रिप्शन विकल्प, टॉपिकल या ओरल फॉर्म शामिल हों, और यह भी बताएं कि डॉक्टर को कब दिखाना चाहिए।
आपका उत्तर स्वाभाविक, सहानुभूतिपूर्ण और चिकित्सकीय रूप से सटीक होना चाहिए।
"""
    else:
        return """You are to act as a highly experienced medical doctor.
          I understand this is for educational purposes only and not a substitute for professional care. 
          You must evaluate the attached image medically. Do not use markdown, formatting symbols, or bullet points. 
          Do not say “as an AI” or refer to your limitations. Speak directly, concisely, and with authority, 
          as if you are consulting a real patient in a clinical setting. 
          Begin directly without any disclaimers or preamble. Do not say “in the image I see” — instead, say “With what I see, I think you have...”. 
          Keep your answer not long ,not more than 3 senctences and  use a single paragraph. After making a quick differential diagnosis,
          suggest appropriate treatments and basic prescription-style remedies, 
          including over-the-counter or prescription options, topical or oral forms, and when to see a doctor.
          Your goal is to sound natural, empathetic, and medically authoritative."""

# Main process function
def process_inputs(audio_file_path, image_filepath, language_code):
    speech_to_text_output = transcribe_with_groq(
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
        audio_file_path=audio_file_path,
        stt_model="whisper-large-v3",
        language_code=language_code
    )

    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=get_system_prompt(language_code) + " " + speech_to_text_output,
            encoded_image=encode_image(image_filepath),
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        doctor_response = "No image provided for me to analyze."

    output_path = "final.mp3"
    text_to_speech_with_gtts(input_text=doctor_response, output_filepath=output_path, language=language_code)
    return speech_to_text_output, doctor_response, output_path

# Language selector UI component
def get_language_dropdown():
    return gr.Dropdown(
        label="Select Language",
        choices=[("English", "en"), ("Hindi", "hi")],
        value="en"
    )

# Gradio Interface Setup
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label=" Speak Your Symptoms"),
        gr.Image(type="filepath", label=" Upload Your Face Image (Optional)"),
        get_language_dropdown()
    ],
    outputs=[
        gr.Textbox(label=" Transcribed Speech"),
        gr.Textbox(label=" Doctor's Diagnosis"),
        gr.Audio(label=" Doctor's Voice", type="filepath")
    ],
    title="AI Doctor with Vision and Voice",
    description="Upload your voice and optional image. Choose language. Get a doctor-style response via voice and text."
)

iface.launch(debug=True)
