#  Rice Grain Instance Segmentation
**Intaligen AI - Computer Vision Task Submission**

[![Open In Colab](https://colab.research.google.com/drive/1belqyjegWNkxrmODsMLhNqCUZwDI577c?usp=sharing)

##  What is this?
This is my submission for the Intaligen AI instance segmentation task. The goal was to take a raw, dense scan of rice grains, separate the touching ones, and output a stylized "paint-stroke" version on a black background.

While working on this, I realized that standard CV algorithms struggle heavily with the dense clumps in the top-left and center of the image. To show my complete thought process, I decided to build and submit **two different pipelines**: one built from scratch using classical OpenCV math, and one using a modern Deep Learning model.

---

##  My Two Approaches

### Approach 1: Marker-Based Watershed 
At first, I tried basic global thresholding, but it kept merging the touching grains together. If I raised the threshold to split the clumps, it deleted the smaller grains. 

To fix this without using AI, I built a custom morphological pipeline from scratch using purely OpenCV:
* I preprocessed the image using **CLAHE** to balance the uneven scanner lighting, followed by **Otsu's Binarization**.
* To map the dense clumps, I computed an Exact **Euclidean Distance Transform**.
* **The Fix:** Instead of a static global threshold, I engineered a **proportional threshold** (`0.3 * max`) on the distance map. This perfectly isolated the thickest core of every single grain—even the touching ones—without erasing the smaller, isolated grains.
* I fed those core points into a **Watershed algorithm** as distinct seeds, which forced the code to accurately separate the touching grains based on pure geometry.

### Approach 2: FastSAM 
While my Watershed code worked great, I know that hardcoding geometric parameters isn't always scalable in the real world. 

So, for my second approach, I wanted to show how I'd solve this in a fast-paced production environment. I used Meta's **Fast Segment Anything Model (FastSAM)** via Ultralytics. Because it actually *understands* what an object is (semantic segmentation), it easily extracted perfect high-resolution masks for overlapping grains with zero-shot accuracy, without needing complex mathematical workarounds.

###  The Stylization (How I matched the reference image)
To get that smooth, non-overlapping look from the `ExpectedOutput.jpeg`, the raw masks from both models needed some post-processing. For every validated mask, my code:
1. **Erodes** the edges slightly (`cv2.erode`) to force a dark, visual gap between touching grains.
2. Applies a **Gaussian Blur** to melt away the jagged pixel edges from the raw scan.
3. Snaps it back to a solid shape using a binary threshold and applies a random vibrant RGB color on a purely black canvas.

---

##  How to look at my code

**The Easiest Way (Google Colab):**
I highly recommend clicking the **"Open in Colab"** badge at the top of this README. I combined both approaches into an interactive Jupyter Notebook where you can read through my thought process and see the images generate inline. *(Note: You just need to upload `InputImage.jpg` to the Colab session first).*

**Running it Locally:**
```bash
# 1. Clone the repo
git clone [https://github.com/YOUR_USERNAME/intaligen-cv-task.git](https://github.com/YOUR_USERNAME/intaligen-cv-task.git)
cd intaligen-cv-task

# 2. Install the libraries
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook notebooks/Rice_Segmentation_Exploration.ipynb
