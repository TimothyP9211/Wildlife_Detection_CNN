"""
Rename all images in a directory to for easier processing

"""

import os

files = sorted(os.listdir("tmp_labels/"))

for i, f in enumerate(files):
    if f.endswith(".jpg") or f.endswith(".png"):
        new_name = f"EPIC_BLACK_BEAR_{i:04d}.jpg"
        os.rename(f"tmp_labels/{f}", f"tmp_labels/{new_name}")
    elif f.endswith(".txt"):
        new_name = f"EPIC_BLACK_BEAR_{i:04d}.txt"
        os.rename(f"tmp_labels/{f}", f"tmp_labels/{new_name}")