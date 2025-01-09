# Automated-classification-of-acoustic-signals
![Automated classification of acoustic signals](https://www.roomeqwizard.com/help/images/spectrogramreflectionswavelet.jpg)
**by Serhii Spitsyn**

# **Description**


## **Purpose of the Project**

# **Stages and progress of the project**

## **Step 1 — Data Pre-processing and Management:**
- Read “large” WAV files and corresponding annotations.
- Extract the desired spectrogram fragments (calls) from the large spectrogram, saving them as 2D arrays (e.g., in .npy or .npz format).
- Similarly, create a dataset of “no-call” spectrograms.
- Save the processed results (arrays and metadata) in the directory structure outcome/....

## **1. General Idea of Data Processing**

### **Data Collection**
- For each `*.wav` file, read the corresponding annotation file `*.Table.1.selections.txt`.  
- Determine how many calls (rows in the `.txt` file) are present in each file and their respective time and frequency bounds.

---

### **Spectrogram**
- Compute a single "large" spectrogram for the entire audio file (e.g., using `signal.spectrogram`) with fixed parameters:
  - `sample_rate` (resample if necessary to a common sampling rate, e.g., 16kHz).
  - `nperseg`, `nfft`, `noverlap`, `window`, etc.
- Extract the spectrogram fragment corresponding to each call.
- Optionally apply frequency limits (e.g., 20 Hz to 1000 Hz) and time constraints based on annotations.

---

### **Size Unification**
- Analyze all annotations to find the maximum call duration and frequency range ("find the longest and widest call").
- For all other calls, apply zero padding (if necessary) or another form of alignment (e.g., time stretching).
- **Goal:** Ensure that each extracted spectrogram has the same size, e.g., `(F × T)`.

---

### **Creating “No-call” Fragments**
- Extract spectrogram fragments from segments that do not overlap with any call (i.e., randomly select time intervals where no annotation exists).
- Ensure that "no-call" fragments have the same size `(F × T)`.
- Ensure that "no-call" fragments cover the same frequency range.

---

## **2. Saving Results**
- For each call (or "no-call" fragment), save the 2D array (spectrogram) in NumPy format.
- Save a minimal set of metadata alongside (in `.csv` format or as separate Python structures). 

The metadata can include:
  - Call type (e.g., Rupe A, B, Moan, etc.),
  - Start/end time in the original file,
  - Original file ID,
  - Any other necessary fields.

---

## **Step 1 - Major Changes and Improvements.**
-	**Adding functionality for processing different folders with data**: iterate through all folders (Rupes A and B, Guttural rupe, Moan, Gray Seal Data Additional), as well as search for *.wav files through glob.glob. 
    - Purpose: Allows you to automate the processing of the entire dataset, not just one "manual" file.
-	**Check audio if multichannel**: Added ensure_mono(samples) function that selects a single channel if the data is stereo.
    - Purpose: Some spectrogram libraries and methods are designed for a one-dimensional signal. This eliminates possible errors and ensures the uniformity of the format.
-	** Clearing/filling small values in the spectrogram**: Moved to the main function compute_spectrogram(...), applied stably for all files.
    - Purpose: Improves the display of the logarithmic scale, avoiding excessively small quantities that can cause problems during visualization or training.
-	**Cut out call and no-call fragments**: Created a function extract_call(...) that, given [t_start, t_end], highlights a subarray of the spectrogram.
    - Purpose: No-call" is necessary to build models with "call presence vs. no call" recognition. This approach makes it possible to balance or expand the sample, reduce false positives in the detector.
-	**Unification of spectrogram size**: A two-step approach has been created:
    - **First pass** — collection of statistics on the maximum number of frequency bins and time bins for all calls (i.e. max_freq_points_global та max_time_points_global).
    - **Second pass** is a direct cut and completion with zeros (np.pad(...)) so that all spectrograms are the same size.
    - Purpose: Most machine learning models (especially CNNs) require the same size of input data (2D "images"). This simplifies the learning process and greatly improves further analysis.
-	** Saving the results in *.npy and generating metadata **: Each cut spectrogram (call or no-call) is stored as a 2D NumPy array (*.npy).
    - Purpose: The *.npy format allows you to quickly load data without loss to train the model (faster and more convenient than working with images). Metadata helps in analyzing results, debugging, and reproducing experiments.
-	**Improved readability and debugging**: The code is structured by functions (compute_spectrogram, extract_call, pick_no_call_segments, etc.). Added debugging messages ([DEBUG]...) that print key information: how long the audio lasts, what speakers are in the annotations, which returns extract_call.
    - Purpose: This makes the project easier to understand and scale, and makes it easier to find errors or analyze why certain spectrograms are cut/not cut.

---




## **Requirements**
To complete the project, you must have:
- Python 3.8+
- Jupyter Notebook


### **Contribution to the project**
Contributions are welcome! You can help improve the project by opening an issue or creating a pull request.

## **Reference**
- [Discovery of sound in the sea](https://dosits.org/galleries/audio-gallery/marine-mammals/pinnipeds/gray-seal/)
- [Grey Seal Halichoerus grypus](https://soundcloud.com/wildlife-sound-recording/grey-seal)
- [The role of vocal learning in call acquisition of wild grey seal pups](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419579/)


## **Contact Information**
- <serhii.spitsyn.ie@gmail.com>
- [GitHub](https://github.com/ShamansIT)