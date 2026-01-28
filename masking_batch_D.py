import os

import cv2
from natsort import natsorted
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import multiprocessing
import datetime
import numpy as np
from masking_functions import *
import re
import colour
import time


def get_time(filename):
    """
    Gets the time from the filenames. You do have to enter the TOD in this function.
    """
    pattern = r'(\d{12})\.png'
    match = re.search(pattern, filename)  # Searches for the pattern in the filenames
    if match:
        time_label = match.group(1)
    else:
        print(f'No match with filename: {filename}')  # If the filenames are formatted differently
        return 0
    dt = datetime.datetime.strptime(time_label, '%Y%m%d%H%M%S')  # This is Year Month Day Hour Minute
    TOD = datetime.datetime(2025, 4, 15, 11, 00)
    delta_seconds = (dt - TOD).total_seconds()
    return delta_seconds


def show_image(name, image):
    """
    Sometimes you want to see the images you're working with. This function is usefull because the resolution
    of the images are so big, this resizes them. The variable term is the name the window will get. The second
    is the image you want to show.
    Example:
        img = cv2.imread(filename)
        show_image("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    pic = cv2.resize(image, (800, 600))
    cv2.imshow(name, pic)


def analyze_image(filename, threshold=130):
    """
    Analyzes the file that is entered. You will have to enter the right control samples and adjust the threshold and
    ratio values so that the right masks are created. Moreover, you need to # the color correction terms if you don't
    use them or enter the right polynomial term if you do.
    """
    red_text = "\033[91m"  # Gives nice red text
    delta_seconds = get_time(filename)  # Gets the time
    transformation_matrix = np.loadtxt('matrix-apr15-2.csv', skiprows=0, delimiter=',')  # Loads the matrix
    control_samples = ['samp12']  # Enter control samples here
    img = cv2.imread(filename)
    if img is None:  # if a image cannot be read
        print(f'{red_text}Fuck you {filename}')  # So you can see which file is not working
        return [None, None, None, None, delta_seconds/3600]  # You're going have to manually delete this from the .csv :)
    img = colour.characterisation.apply_matrix_colour_correction_Cheung2004(
        img / 255, transformation_matrix, 14)  # Enter the right polynomial term
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)  # Makes the images floats
    background_mask = img  # Make copies to use
    bloodstain_mask = img
    outer_bloodstain_mask = img
    inner_bloodstain_mask = img
    bloodstain_mask2, outer_bloodstain_mask2, inner_bloodstain_mask2, background_mask2 = None, None, None, None
    for sample in control_samples:
        if sample in filename:
            break
        else:  # Enter the right threshold and ratio values
            blur = make_blur(img)
            background_mask2, background_mask = divide_mask(img, blur, threshold=0.32, background=True)
            bloodstain_mask2, bloodstain_mask = make_bloodstain_mask(img, blur, threshold=0.32)
            outer_bloodstain_mask2, outer_bloodstain_mask = divide_mask(img, blur, threshold=0.32, ratio=0.7)
            inner_bloodstain_mask2, inner_bloodstain_mask = divide_mask(img, blur, threshold=0.32, ratio=0.7, inner=True)
    results = [cv2.mean(bloodstain_mask, mask=bloodstain_mask2),
               cv2.mean(outer_bloodstain_mask, mask=outer_bloodstain_mask2),
               cv2.mean(inner_bloodstain_mask, mask=inner_bloodstain_mask2),
               cv2.mean(background_mask, mask=background_mask2),
               delta_seconds / 3600]
    return results


def do_batch_analysis(chosen_directory):
    """
    Does the batch analysis for a chosen directory.
    """
    filenames = [os.path.join(chosen_directory, file) for file in natsorted(os.listdir(chosen_directory))
                 if file.lower().endswith(('png', 'jpg'))]
    total_images = len(filenames)
    print(f"Starting batch analysis for {total_images} images...")
    start_time = time.time()
    results = []
    batch_size = 10
    with multiprocessing.Pool() as pool:
        for i in range(0, total_images, batch_size):
            batch_files = filenames[i:i + batch_size]
            batch_results = pool.map(analyze_image, batch_files)
            batch_end = time.time()
            results.extend(batch_results)
            elapsed = batch_end - start_time
            avg_time_per_image = elapsed / (i + batch_size)
            estimated_total_time = avg_time_per_image * total_images
            remaining_time = estimated_total_time - elapsed
            print(f"Processed {min(i + batch_size, total_images)}/{total_images} images. ",
                  f"Elapsed: {elapsed:.2f}s, Estimated Remaining: {remaining_time / 3600:.2f}h")
    total_time = time.time() - start_time
    print(f"Batch analysis completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes).")
    bloodstain_val, outer_bloodstain_val, inner_bloodstain_val, background_val, delta_hours = zip(*results)
    return bloodstain_val, outer_bloodstain_val, inner_bloodstain_val, background_val, delta_hours, filenames


def open_dir():
    """
    Gets the directory
    """
    directory_name = filedialog.askdirectory()
    root.destroy()
    return directory_name


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # So that no window remains after selecting a directory
    directory = filedialog.askdirectory()
    # directory = '/home/afmlab/Documents/Serafine/data aq/2022-12-07 13:43:08.512471' # If you wanna do it manually
    if directory:  # Gets all the variables and puts them in a Df
        bloodstain, outer_bloodstain, inner_bloodstain, background, time, files = do_batch_analysis(directory)
        df = pd.DataFrame({'Filename': files, 'Bloodstain': bloodstain, 'Outer bloodstain': outer_bloodstain,
                           'Inner bloodstain': inner_bloodstain,
                           'Background': background, 'Time (hours)': time})
        df['Bloodstain'] = df['Bloodstain'].apply(np.array)
        df['Background'] = df['Background'].apply(np.array)
        df['Outer bloodstain'] = df['Outer bloodstain'].apply(np.array)
        df['Inner bloodstain'] = df['Inner bloodstain'].apply(np.array)
        df.to_csv('red_values-15-04.csv', index=False)
