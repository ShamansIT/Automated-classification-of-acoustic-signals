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

## **3. Directory Structure**
The processed results (arrays and metadata) are saved in the following structure:

```
**outcome/**
├── processed_calls/ │ 
├── call_001.npy │ 
├── call_002.npy │ 
└── ... ├── processed_no_calls/ │ 
├── no_call_001.npy │ 
├── no_call_002.npy │ 
└── ... 
└── metadata.csv
```

---

### **Additional Notes:**
- Ensure consistency in sample rates and spectrogram dimensions across all files to avoid shape mismatches.
- Consider verifying the integrity of each file after saving to ensure data accuracy.





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