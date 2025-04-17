from gtts import gTTS
import os
import platform
import subprocess

def text_to_speech_with_gtts(input_text, output_filepath="gtts_output_file.mp3"):
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts.save(output_filepath)

    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(["afplay", output_filepath])
        elif os_name == "Windows":
            subprocess.run(["ffplay", "-nodisp", "-autoexit", output_filepath], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os_name == "Linux":
            subprocess.run(["aplay", output_filepath])
    except Exception as e:
        print(f"Audio error: {e}")
