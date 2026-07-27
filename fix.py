with open("benchmark_wav.py", "r") as f:
    text = f.read()

text = text.replace("import soundfile as sf", "import soundfile as sf\nimport numpy as np")
text = text.replace("        audio_np = audio_np[:, np.newaxis]", "        audio_np = audio_np[:, np.newaxis]")

with open("benchmark_wav.py", "w") as f:
    f.write(text)
