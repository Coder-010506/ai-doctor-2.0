# voice_of_the_doctor.py
import platform
import subprocess
from gtts import gTTS
from pydub import AudioSegment

def text_to_speech_with_gtts(input_text, output_filepath, language="en"):
    try:
        tts = gTTS(text=input_text, lang=language, slow=False)
        tts.save(output_filepath)
    except Exception as e:
        print(f" Error generating audio: {e}")

def play_audio_file(filepath):
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(["afplay", filepath])
        elif os_name == "Windows":
            subprocess.run(["powershell", "-c", f'(New-Object Media.SoundPlayer \"{filepath}\").PlaySync();'], shell=True)
        elif os_name == "Linux":
            subprocess.run(["aplay", filepath])
        else:
            print(" Unsupported OS")
    except Exception as e:
        print(f" Playback error: {e}")
