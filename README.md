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

## **Step 2 - Model Training.**

### Experiment Objective
- Train a model to classify acoustic signals (spectrograms) using various vocalization classes.  
- Evaluate the accuracy and quality of recognizing individual classes using metrics such as the confusion matrix and classification report.

### Settings and Methods
- **Model Architecture**: SimpleCNN (2 convolutional layers with ReLU + MaxPool, AdaptiveAvgPool2D, 2 linear layers).  
- **Training Parameters**:  
  - Batch Size: 8  
  - EPOCHS: 5  
  - Learning Rate: 1e-3  
- **Loss and Optimization**: Adam + CrossEntropyLoss.  
- **Dataset Splitting**:  
  - 70% train  
  - 15% validation  
  - 15% test  
- **Hardware**: Used `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

### Brief Result Summary
- **Overall Accuracy (Test Accuracy):** ~0.48.  
- **Best Recognized Class** — G rupe (recall: 0.87), but with low precision (0.42) due to the model classifying many other samples as G rupe.  
- **Rupe A** shows the second-best performance (precision: 0.61, recall: 0.73).  
- For the remaining classes, **precision = 0.0**, indicating an almost total lack of correct predictions for them.

### Error Analysis
- The model often confuses **G rupe** with **Rupe A**, and it confuses nearly all other classes with G rupe as well.  
- Possible causes: data imbalance, signal similarities, and a limited architecture.

### Next Steps
1. **Improving dataset balance** — collect or augment data for rare classes.  
2. **Exploring deeper models** or pretrained networks for audio/spectrogram tasks.  
3. **Augmentation experiments** (time shift, specaugment, adding noise).  
4. **Hyperparameter optimization** (adjust LR, increase epochs, regularization).  
5. **Additional analysis** of characteristic frequencies or time-based features for specific vocalizations.

In **Step 2** obtained a basic CNN model which achieved ~48% overall accuracy but strongly favors two main classes. Going forward, efforts should focus on ensuring better discrimination among more classes through dataset balancing, finding an improved architecture, and more in-depth tuning.

## Analytical Findings Step 2

### General Overview
The model performs best in recognizing the classes **G rupe** and **Rupe A**. For **G rupe**, we observe a fairly high recall (0.87), whereas **Rupe A** exhibits relatively balanced precision (0.61) and recall (0.73).  
Most of the other classes are practically not recognized (precision and recall = 0.0). This indicates a strong mismatch (or bias) of the model towards certain specific classes.

### Confusion Matrix Details
- A large number of examples from various classes are mistakenly classified as **G rupe**.  
- For instance, the **Moan** class (row 3, given zero-based indexing in the original confusion matrix) has 62 examples, but in 61 cases the model predicts **G rupe**, and only 1 case is identified as **Rupe A**.  
- This explains the high recall value for **G rupe** (0.87): almost all true G rupe samples are recognized as G rupe, but many other classes also get “pulled in” as G rupe.  
- The **Rupe A** class (78 examples) is correctly classified as Rupe A in 57 cases, while 21 cases are classified as G rupe.  
- Between these two classes (G rupe and Rupe A), there is a relative “competition”: the model often confuses G rupe with Rupe A.  
- Other classes (e.g., Guttural rupe, Rupe B, Rupe C, Trrot, Type 4 A, etc.) are almost always confused with G rupe or occasionally with Rupe A.  
- This indicates the model’s insufficient ability to distinguish smaller or rarer classes, likely due to imbalanced training data or shortcomings in architecture/hyperparameters.

### Key Metrics
- **Precision (0.42) and recall (0.87) for G rupe** indicate that while the model successfully “captures” most G rupe samples, it also “captures” many foreign samples — causing precision to drop (many false positives).  
- For **Rupe A** (precision: 0.61, recall: 0.73), the results are better than for G rupe in terms of balance, yet the overall contribution to the model’s accuracy remains limited.  
- For the rest of the classes, **precision and recall = 0.0**, because the model either does not predict them at all or predicts them in only a few instances, leading to low metrics.

### Possible Reasons for Low Metrics in Other Classes
1. **Dataset Imbalance**: If most examples belong to a few “major” classes (G rupe, Rupe A), the model may “learn” to ignore rarer classes.  
2. **Insufficient Training Examples** in rare classes and/or poor-quality spectrograms (noise, overlap, incomplete data).  
3. **SimpleCNN Architecture**: It may lack the “capacity” to distinguish classes with similar acoustic characteristics.  
4. **Hyperparameters** (number of epochs, learning rate, batch size) might not be optimal.

### Recommendations for Improvement
- Collect more **balanced data** or apply balancing methods (oversampling, undersampling, data augmentation).  
- Use **more complex models** (e.g., ResNet, EfficientNet) or **pre-trained embeddings** (e.g., audio embeddings) and then fine-tune.  
- Perform **fine-tuning of hyperparameters**: increase the number of epochs, adjust the learning rate, batch size, etc.  
- Expand **augmentations**: vary noise, adjust volume, apply time masking or frequency masking (spectrogram augmentations).  
- **Analyze the spectrograms** further: try different Mel-spectrogram parameters (number of mel-bands, window size, overlap) to better highlight characteristic features.

---

# Step 3 – Refine (Research Phase)

This section describes the **third phase** of our project, aimed at **refining** previous results. Since this is a **research project**, the evaluation will be based on a **research approach** rather than simply achieving the "perfect" classifier.

---

## 1. Overall Goal of "Refine"

- **Improve** the process of spectrogram extraction and data preparation (e.g., increase `nfft`, `noverlap`, `fmax`, apply padding, etc.).
- **Adjust** models and training strategies (CNN or Transfer Learning parameters, learning rate, number of epochs, regularization techniques).
- **Conduct experiments** with different hyperparameters, analyze results, and document observations.

---

## 2. Key Changes in the Refine Phase

1. **Spectrogram Parameters**:
   - Increased `nfft` (e.g., to 8192) for higher frequency resolution.
   - Increased `noverlap` (e.g., to 2048) to achieve smoother transitions between time windows.
   - Extended the frequency range (`fmax`) to 2000 Hz, allowing for detection of higher harmonics in seal vocalizations.

2. **Spectrogram Padding**:
   - Implemented the `pad_spectrogram()` function so that all extracted fragments (calls and no-calls) have the **same** size. This solves the issue of "different-length" spectrograms and allows proper batch formation in the PyTorch DataLoader.

3. **Model and Training Configuration**:
   - Expanded or modified the CNN architecture (SimpleCNN) or applied Transfer Learning (ResNet) with a fixed first layer for 1-channel spectrograms.
   - Conducted experiments with different `batch_size`, `learning_rate`, and `EPOCHS`.  
   - Evaluated accuracy, confusion matrix, and precision/recall/F1-score.

4. **Analysis of "Rare" Classes**:
   - Found that some classes have very few examples (support < 5), leading to 0.0 in precision/recall metrics.  
   - To address this, we considered strategies such as data augmentation (oversampling, data augmentation) or merging some classes.

---

## 3. Research Approach

Since we focus on **research**, the main steps included:

1. **Parameter Verification and Documentation**:
   - To what extent does increasing `nfft` impact the spectrogram computation time?
   - How does `noverlap` affect temporal resolution?  
2. **Experimental Validation**:
   - Comparing results (accuracy, F1) before and after increasing `fmax`.
   - Analyzing whether adding `TIME_BUFFER` or splitting WAV files into 60-second fragments improves results.
   - Running multiple training variations (different `lr`, `batch_size`) and comparing metrics.
3. **Maintaining Logs**:
   - Stored each experiment in CSV or notes: the hyperparameters and the resulting metrics.
   - Documented hypotheses (e.g., “increase `nfft` — will F1-score improve?”).

---

## 4. Key Results

1. **Accuracy** slightly increased (e.g., from ~49% to ~53%).
2. **Some classes** (notably “Rupe A” or its equivalent) show significantly improved recall, reaching 0.90.  
3. **Certain classes** with low support (1–5 samples) still have 0.0 precision/recall due to insufficient data.  
4. **Weighted F1-score** also improved, but the dataset still needs better balance or rare class handling.

---

## 5. Challenges and Future Actions

1. **Data Imbalance**  
   - Track whether there are too few samples for some vocalization types. Oversampling or **artificial** data augmentation (SpecAugment, time shifting, frequency masking) may be necessary.

2. **Model Architecture Variations**  
   - Try ResNet18 vs. SimpleCNN.  
   - Test whether ResNet50 or EfficientNet-B0/B1 yields better metrics.

3. **Hyperparameter Optimization**  
   - Implement a **scheduler** (e.g., `ReduceLROnPlateau` or `StepLR`).  
   - Increase the number of epochs (e.g., 20–30) if overfitting is not observed.

4. **Further Refinements in "Refine"**  
   - Adjust `MAX_FREQ_DIM` and `MAX_TIME_DIM` for padding to capture the real boundaries of the signal.  
   - Experiment with **shorter** or **longer** call durations (SEGMENT_DURATION) to see if it impacts accuracy.

---

## 6. Step 3 – Refine

In the **"Refine" phase**, we achieved:
- **Higher-quality** data preparation (higher `nfft`, wider `fmax`, padding).  
- **Experiments** with models and hyperparameters, comparing new metrics with the old ones.  
- Despite the continued lack of data for some classes (0.0 precision/recall), the overall results **improved** (average accuracy).

Since this is a **research** project, the **main value** lies in thorough analysis and experiment logging. Future plans include:
- Additional augmentation for rare classes.  
- Testing other architectures and schedulers.  
- Potential merging of very small classes for more reliable statistics.

**Step 3 – Refine** brings us closer to a **deeper** understanding of how spectrogram and model parameter settings affect the final classification. The results help outline further experiments to improve the seal vocalization detector/classifier.




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
- [Fundamentals of spectral analysis](https://www.ele.uva.es/~lourdes/docencia/Master_IE/Fundamentals_of_Spectrum_Analysis%5B2%5D.pdf)
- [Spectrum Analysis Basics](https://www.keysight.com/blogs/en/tech/rfmw/2020/05/01/spectrum-analysis-basics-part-1-what-is-a-spectrum-analyzer)
- [Machine learning in spectral domain](https://www.nature.com/articles/s41467-021-21481-0)
- [Deep learning in spectral analysis: Modeling and imaging](https://www.sciencedirect.com/science/article/abs/pii/S0165993624000943)
- [Machine learning enhanced spectroscopic analysis](https://pubs.rsc.org/en/content/articlehtml/2022/dd/d1dd00027f)


## **Contact Information**
- <serhii.spitsyn.ie@gmail.com>
- [GitHub](https://github.com/ShamansIT)