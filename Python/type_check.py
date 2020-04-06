from PIL import Image
import os

base = "../data/ILSVRC2012_img_val/"

raw_walk_gen = os.walk(base + "raw/", topdown=True)
_, _, files_raw = next(raw_walk_gen)
files = [f for f in files_raw if f != ".gitignore"] # Remove .gitignore
sorted_files = sorted(files, key=lambda item: int(item[18:23]))

modes = {}

for _, name in enumerate(sorted_files):
    orig_img = Image.open(base + "raw/" + name)
    mode = orig_img.mode
    if mode in modes:
        modes[mode] += 1
    else:
        modes[mode] = 1

print(modes)