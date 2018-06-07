from PIL import Image
import os
files = os.listdir()
files = [f for f in files if 'jpg' in f or 'jpeg' in f or 'png' in f]

badfiles = []
for f in files:
    try:
        img = Image.open(f)
    except:
        badfiles.append(f)

for bf in badfiles:
    os.remove(bf)