# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 11:45:02 2025

@author: ella
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
import datetime
import tkinter as tk
from tkinter import filedialog
from natsort import natsorted
from multiprocessing import Pool
from masking_functions import make_blur, make_bloodstain_mask, divide_mask
import colour


# --------------------------- FUNCTIES ---------------------------

def get_time(filename):
    """Haal de tijd op uit de filename (format YYYYMMDDHHMM.png)"""
    pattern = r'(\d{12})\.png'
    match = re.search(pattern, filename)
    if not match:
        print(f'No match with filename: {filename}')
        return 0
    dt = datetime.datetime.strptime(match.group(1), '%Y%m%d%H%M')
    TOD = datetime.datetime(2025, 1, 30, 15, 0)
    return (dt - TOD).total_seconds()


def show_image(name, image):
    """Toon een afbeelding in een 800x600 venster"""
    pic = cv2.resize(image, (800, 600))
    cv2.imshow(name, pic)


def analyze_image(filename, threshold=80, matrix_path=None, poly_degree=3):
    """Analyseer één afbeelding: kleurcorrectie + masking"""
    red_text = "\033[91m"
    delta_hours = get_time(filename) / 3600

    # Lees de correctiematrix
    transformation_matrix = np.loadtxt(matrix_path, delimiter=',')

    control_samples = ['samp1', 'samp12']
    img = cv2.imread(filename)
    if img is None:
        print(f'{red_text}Image cannot be read: {filename}')
        return [None, None, None, None, delta_hours]

    # Kleurcorrectie
    img_float = np.float32(img) / 255
    corrected_img = colour.characterisation.apply_matrix_colour_correction_Cheung2004(
        img_float, transformation_matrix, poly_degree
    )
    img = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)

    # Maak masks
    background_mask = img
    bloodstain_mask = img
    outer_bloodstain_mask = img
    inner_bloodstain_mask = img
    bloodstain_mask2 = outer_bloodstain_mask2 = inner_bloodstain_mask2 = background_mask2 = None

    for sample in control_samples:
        if sample in filename:
            break
        else:
            blur = make_blur(img)
            background_mask2, background_mask = divide_mask(img, blur, threshold=0.35, background=True)
            bloodstain_mask2, bloodstain_mask = make_bloodstain_mask(img, blur, threshold=0.35)
            outer_bloodstain_mask2, outer_bloodstain_mask = divide_mask(img, blur, threshold=0.35, ratio=0.7)
            inner_bloodstain_mask2, inner_bloodstain_mask = divide_mask(img, blur, threshold=0.35, ratio=0.7, inner=True)

    return [
        cv2.mean(bloodstain_mask, mask=bloodstain_mask2),
        cv2.mean(outer_bloodstain_mask, mask=outer_bloodstain_mask2),
        cv2.mean(inner_bloodstain_mask, mask=inner_bloodstain_mask2),
        cv2.mean(background_mask, mask=background_mask2),
        delta_hours
    ]


def do_batch_analysis(chosen_directory, matrix_path):
    """Voer batch-analyse uit op alle afbeeldingen in een map"""
    filenames = [
        os.path.join(chosen_directory, f) for f in natsorted(os.listdir(chosen_directory))
        if f.lower().endswith(('png', 'jpg'))
    ]

    with Pool() as pool:
        results = pool.starmap(analyze_image, [(f, 80, matrix_path) for f in filenames])

    bloodstain_val, outer_bloodstain_val, inner_bloodstain_val, background_val, delta_hours = zip(*results)
    return bloodstain_val, outer_bloodstain_val, inner_bloodstain_val, background_val, delta_hours, filenames


def open_dir():
    """Open een directory-venster om een map te kiezen"""
    root = tk.Tk()
    root.withdraw()
    directory_name = filedialog.askdirectory()
    root.destroy()
    return directory_name


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    matrix_path = r"C:\Users\ella\OneDrive - De Haagse Hogeschool\Stage 1\Masking Code\matrix-apr15-low-1.csv"
    directory = open_dir()

    if directory:
        bloodstain, outer_bloodstain, inner_bloodstain, background, time_hours, files = do_batch_analysis(directory, matrix_path)

        df = pd.DataFrame({
            'Filename': files,
            'Bloodstain': bloodstain,
            'Outer bloodstain': outer_bloodstain,
            'Inner bloodstain': inner_bloodstain,
            'Background': background,
            'Time (hours)': time_hours
        })

        df['Bloodstain'] = df['Bloodstain'].apply(np.array)
        df['Outer bloodstain'] = df['Outer bloodstain'].apply(np.array)
        df['Inner bloodstain'] = df['Inner bloodstain'].apply(np.array)
        df['Background'] = df['Background'].apply(np.array)

        df.to_csv('red-values.csv', index=False)
        print("Batch-analyse voltooid! CSV opgeslagen als 'red-values.csv'.")
