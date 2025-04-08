# 🦷 **DentalVision: Automated Teeth Segmentation for Dental Diagnostics** 🦷

DentalVision is an advanced tool designed to **automatically segment teeth** in **dental X-ray images** using cutting-edge **deep learning techniques**. The tool leverages **Vision Transformers (ViT)**, **ResNet-34**, and a **U-Net-style decoder** to provide fast and accurate segmentation results, enhancing **diagnostic workflows** in dentistry.


## 🚀 **Project Overview**

DentalVision is a powerful tool that automates the process of teeth segmentation in dental X-ray images, leveraging deep learning models like **Vision Transformer (ViT)** and **ResNet-34** (fine-tuned). The tool enhances diagnostic accuracy, reduces human error, and streamlines the diagnostic workflow for dental professionals.

- **Key features**:
  - **ViT-based Encoder + U-Net Decoder** for teeth segmentation.
  - **Fine-tuned ResNet-34** model for improved segmentation accuracy.
  - **Image Processing** Traditional Image pricessing techniques.
  - **Real-time results** through the interactive **Streamlit** web app.

---

## ⚙️ **Installation**

### 🔧 **Prerequisites**
Before running the project, ensure the following dependencies are installed:

- Python 3.6 or higher
- PyTorch
- TensorFlow (for ResNet-34 model)
- Streamlit
- Hugging Face's Transformers library
- Git LFS (Large File Storage) for model files
- Kaggle Dataset Link: https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images

### 📥 **Installation Steps**
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/DentalVision.git
    ```

2. Navigate to the project directory:
    ```bash
    cd DentalVision
    ```

3. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. **Git LFS**: Ensure **Git Large File Storage** is installed to handle large model files (e.g., `.pth`, `.hdf5`):
    - Install Git LFS:  
    ```bash
    git lfs install
    ```

5. Run the **Streamlit app**:
    ```bash
    streamlit Dental Vision.py
    ```

---

## 📜 **Usage**

### 🦷 **Web Interface**
- Visit the deployed **Streamlit** app at:  
  [DentalVision Streamlit App](https://dentalvision.streamlit.app)

- **Steps**:
  1. Upload a dental X-ray image.
  2. View the segmentation results:
     - **Result 1**: Traditional image processing.
     - **Result 2**: ViT-based segmentation.
     - **Result 3**: CNN-based segmentation (fine-tuned ResNet-34).

---

## 🧠 Model Training
Vision Transformer (ViT) was used for feature extraction and segmentation, leveraging transformer-based self-attention to capture long-range dependencies within the image.

The model was trained with a binary cross-entropy loss function and Adam optimizer, yielding IoU of 0.82 and Dice coefficient of 0.89.

## 🔄 Model Fine-Tuning
The ResNet-34 model was fine-tuned for teeth segmentation. Using transfer learning, we adapted the pre-trained model to the dental X-ray dataset, significantly improving performance compared to traditional methods.

## 📊 Results
The model achieved strong segmentation results:

IoU: 0.82 (Intersection over Union)

Dice Coefficient: 0.89

The results show that the ViT-based encoder with a U-Net decoder provides highly accurate segmentation masks. Furthermore, fine-tuning ResNet-34 improved the performance of the segmentation model.

- Resnet 34 Fine Tuning Notebook: https://drive.google.com/file/d/1gUdX9RXpKMK6VDV4OqPK45mOnpbFuFkR/view?usp=sharing
- 
Trained Models Link:

Resnet 34: https://drive.google.com/file/d/1d5Yka9qb4Rd22PWxsGgfjOnx_wNJssLQ/view?usp=sharing

VIT model : https://drive.google.com/file/d/1FbZ77_AmIy0_DFM1g5XC_0wZDEJlCFTS/view?usp=sharing

## 🤝 Team Members:
Abhishek Kumar Singh (M23CSA503) 

Ankit Kumar Chauhan (M23CSA509) 

Himani (M23CSA516) 

Rishabh Sharma (M23CSA523)
