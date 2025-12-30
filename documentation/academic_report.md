# Academic Report: Audio Deepfake Detection System

## 1. Evaluation and Explainability

### 1.1 Why Spectrogram-Based Learning Works
The human auditory system processes sound by analyzing its frequency content over time. Spectrograms, particularly **Mel-Spectrograms**, visually represent this time-frequency information, making them highly effective for audio processing tasks.

| Feature | Description | Relevance to Deepfake Detection |
| :--- | :--- | :--- |
| **Time Axis** | Represents the temporal evolution of the audio signal. | Captures the dynamic changes in speech, which can be unnatural or overly smooth in synthetic audio. |
| **Frequency Axis (Mel Scale)** | Represents the distribution of energy across different frequencies, mimicking human perception. | Highlights spectral artifacts, such as high-frequency noise or band-limited energy, that are characteristic of specific Text-to-Speech (TTS) or Voice Conversion (VC) algorithms. |
| **Intensity (Color)** | Represents the magnitude (energy) of the signal at a given time and frequency. | Reveals subtle energy fluctuations and inconsistencies that differentiate genuine (bona fide) speech from machine-generated spoofs. |

Deepfake generation processes, especially those involving vocoders (like WaveNet or WaveGlow), often introduce subtle, non-linear distortions in the phase and magnitude of the frequency components. These distortions are often invisible in the raw waveform but manifest as distinct, detectable patterns or "fingerprints" in the Mel-Spectrogram image.

### 1.2 Why a 2D CNN is Suitable
A **2D Convolutional Neural Network (CNN)** is the ideal choice for processing Mel-Spectrograms because it treats the spectrogram as a two-dimensional image, allowing it to exploit the spatial correlation inherent in the data.

| CNN Advantage | Application to Spectrograms |
| :--- | :--- |
| **Local Feature Extraction** | Small convolutional kernels (e.g., 3x3) can detect localized patterns in time and frequency, such as formants, pitch contours, or spectral ripples. |
| **Parameter Sharing** | The same filter is applied across the entire spectrogram, making the model robust to variations in the position of a specific artifact (e.g., a spectral glitch appearing slightly earlier or later). |
| **Hierarchical Feature Learning** | Deeper layers combine low-level features (edges, textures) into high-level, abstract representations that are highly discriminative for classifying "real" vs. "fake" acoustic characteristics. |

### 1.3 Key Evaluation Metrics
While standard metrics like **Accuracy**, **Precision**, and **Recall** are important, the primary metric for the ASVspoof challenge is the **Equal Error Rate (EER)**.

| Metric | Definition | Significance |
| :--- | :--- | :--- |
| **Accuracy** | Overall proportion of correct predictions. | General measure of model performance. |
| **EER (Equal Error Rate)** | The point where the False Acceptance Rate (FAR) equals the False Rejection Rate (FRR). | The standard, single-value metric for anti-spoofing systems, representing the trade-off between mistakenly accepting a fake (FAR) and mistakenly rejecting a real (FRR). |
| **ROC Curve** | Plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. | Visualizes the model's performance across all possible classification thresholds. |
| **Classification Report** | Provides Precision, Recall, and F1-Score for each class (Real/Fake). | Essential for understanding the model's performance on the minority class (often the bona fide samples in the ASVspoof dataset). |

## 2. Bonus Content: Advanced Considerations

### 2.1 Theoretical Comparison: CNN vs. RawNet2
| Feature | CNN (Spectrogram-based) | RawNet2 (Raw Waveform-based) |
| :--- | :--- | :--- |
| **Input** | 2D Mel-Spectrogram image. | 1D raw audio waveform. |
| **Core Architecture** | 2D Convolutional Layers. | 1D Convolutional Layers (often deep and residual). |
| **Feature Extraction** | Explicitly extracts time-frequency features (Mel-Spectrogram) before the model. | Implicitly learns time-frequency features directly from the raw signal. |
| **Advantages** | Computationally efficient; leverages well-established image processing techniques; features are human-interpretable. | Avoids information loss from feature engineering; potentially captures phase information missed by magnitude-only spectrograms. |
| **Disadvantages** | Discards phase information; performance is highly dependent on the quality of the Mel-Spectrogram parameters. | High computational cost; requires very deep networks to learn meaningful features; less interpretable. |
| **Suitability** | Excellent for detecting spectral artifacts (TTS/VC). | Strong performance on replay attacks and subtle waveform distortions. |

### 2.2 Noise-Robust Preprocessing
To improve the model's robustness against environmental noise (a common challenge in real-world deployment), the following techniques can be theoretically applied:

1.  **Feature-level Normalization**: Applying **Mean and Variance Normalization** across the entire dataset or per-utterance to mitigate channel and noise variations.
2.  **Noise Augmentation**: During training, randomly adding small amounts of **Additive White Gaussian Noise (AWGN)** or environmental noise (e.g., from the [DEMAND database](https://www.kaggle.com/datasets/chrisfilo/demand-database)) to the raw audio before feature extraction.
3.  **Feature Enhancement**: Using techniques like **Spectral Subtraction** or **Wiener Filtering** to reduce stationary noise components in the spectrogram before feeding it to the CNN.

### 2.3 Optimizing Inference Speed
For a production-ready application, inference speed is critical.

1.  **Model Quantization**: Converting the model's weights and activations from 32-bit floating point to 16-bit (half-precision) or 8-bit integers. This significantly reduces model size and memory bandwidth, leading to faster CPU/GPU inference with minimal loss in accuracy.
2.  **Model Pruning and Distillation**: Removing redundant connections (pruning) or training a smaller "student" model to mimic the behavior of a larger "teacher" model (distillation).
3.  **Framework Optimization**: Using optimized inference engines like **TorchScript** (for PyTorch) or **ONNX Runtime**. These tools compile the model into an optimized graph representation, allowing for faster execution on various hardware platforms.
4.  **Efficient Feature Extraction**: Pre-calculating and caching the Mel-Spectrograms for static datasets. For real-time inference, ensuring the `librosa` feature extraction is performed efficiently, potentially using a lower sampling rate (e.g., 8kHz instead of 16kHz) if performance allows.
