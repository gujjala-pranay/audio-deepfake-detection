# Guide to Large-Scale Audio Deepfake Datasets

This guide provides information and instructions for using large-scale datasets, specifically the ASVspoof challenges, to train and evaluate your Audio Deepfake Detection model. Utilizing these datasets will allow for more robust model training and better generalization capabilities.

## 1. Understanding Large Datasets

Large datasets like ASVspoof 2019 and ASVspoof 2021 contain a vast number of real (bona fide) and spoofed (fake) audio samples. These datasets are crucial for developing high-performance deepfake detection systems due to their diversity in attack types, recording conditions, and speaker variations. However, they also require significant computational resources (CPU, GPU, RAM, storage) for processing and training.

## 2. Official Dataset Download Links

It is highly recommended to download these datasets from their official sources to ensure data integrity and compliance with their respective licenses. Note that these files are very large (several gigabytes) and may take a considerable amount of time to download.

### 2.1 ASVspoof 2019

The ASVspoof 2019 dataset is divided into Logical Access (LA) and Physical Access (PA) scenarios.

| Dataset Partition | Size | Download Link |
| :---------------- | :--- | :------------ |
| **LA.zip** (Logical Access) | ~7.1 GB | [Download LA.zip](https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip) |
| **PA.zip** (Physical Access) | ~16.4 GB | [Download PA.zip](https://datashare.ed.ac.uk/bitstream/handle/10283/3336/PA.zip) |

*Source: [University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336)*

### 2.2 ASVspoof 2021 (Deepfake Task)

The ASVspoof 2021 challenge includes a dedicated Deepfake (DF) task. The evaluation set for this task is provided in multiple parts.

| Dataset Partition | Size | Download Link |
| :---------------- | :--- | :------------ |
| **ASVspoof2021_DF_eval_part00.tar.gz** | ~8.6 GB | [Download Part 00](https://zenodo.org/record/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz?download=1) |
| **ASVspoof2021_DF_eval_part01.tar.gz** | ~8.6 GB | [Download Part 01](https://zenodo.org/record/4835108/files/ASVspoof2021_DF_eval_part01.tar.gz?download=1) |
| **ASVspoof2021_DF_eval_part02.tar.gz** | ~8.6 GB | [Download Part 02](https://zenodo.org/record/4835108/files/ASVspoof2021_DF_eval_part02.tar.gz?download=1) |
| **ASVspoof2021_DF_eval_part03.tar.gz** | ~8.6 GB | [Download Part 03](https://zenodo.org/record/4835108/files/ASVspoof2021_DF_eval_part03.tar.gz?download=1) |

*Source: [Zenodo](https://zenodo.org/records/4835108)*

## 3. Downloading and Extracting the Data

1.  **Download**: Use `wget` or your browser to download the desired `.zip` or `.tar.gz` files to your `audio-deepfake-detection/data/` directory.
    ```bash
    cd audio-deepfake-detection/data/
    wget <download_link_for_LA.zip>
    wget <download_link_for_PA.zip>
    # For ASVspoof 2021, download all parts
    ```

2.  **Extract**: Extract the contents of the downloaded archives.
    ```bash
    unzip LA.zip
    tar -xzf ASVspoof2021_DF_eval_part00.tar.gz
    # Repeat for all parts
    ```
    *Ensure you have enough disk space (tens of gigabytes) before extracting.*

## 4. Integrating with Your Project

To train your model on the full dataset, you need to modify the `utils.py` file to remove the `limit` parameter that was previously set for medium-sized datasets.

1.  **Open `audio-deepfake-detection/utils.py`**.

2.  **Locate the `AudioDataset` class constructor**:
    Find the line that defines the `__init__` method:
    ```python
    def __init__(self, protocol_file, data_dir, transform=None, max_len=64000, limit=10000):
    ```

3.  **Remove or comment out the `limit` parameter and its usage**:
    Change the line to:
    ```python
    def __init__(self, protocol_file, data_dir, transform=None, max_len=64000):
    ```
    And remove or comment out the following lines within the `__init__` method:
    ```python
                # Use a medium dataset size
                if limit and len(lines) > limit:
                    lines = lines[:limit]
    ```
    This will ensure that the `AudioDataset` loads all available samples from your protocol file.

## 5. Training Considerations for Large Datasets

-   **Hardware**: Training on full ASVspoof datasets typically requires powerful GPUs with substantial VRAM (e.g., 16GB or more) and a large amount of RAM.
-   **Training Time**: Expect training to take hours or even days, depending on your hardware and model complexity.
-   **Batch Size**: You may need to reduce the `batch_size` in your training script (`src/train.py`) to fit the data into GPU memory.
-   **Data Loading**: Consider implementing more advanced data loading techniques (e.g., PyTorch `DataLoader` with `num_workers > 0`) to speed up data fetching, though this also increases RAM usage.

By following these steps, you can leverage the full potential of these large datasets to build a highly accurate audio deepfake detection system.
