
import tempfile
import os
import whisper

model = whisper.load_model("base")

def transcribe_audio(audio_file):
    suffix = os.path.splitext(audio_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    result = model.transcribe(tmp_path)
    os.remove(tmp_path)
    return result["text"]
