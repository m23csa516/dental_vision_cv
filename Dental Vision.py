
import io
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tensorflow import keras
import time
import base64
from io import BytesIO
import os
from PIL import Image
# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from transformers import ViTModel, ViTConfig
import torch.nn.functional as F
import gdown
col1, col2, col3 = st.columns(3)
# Define a custom neural network that combines Vision Transformer (ViT) and U-Net-style decoder
class ViTUNet(nn.Module):
    def __init__(self):
        super(ViTUNet, self).__init__()

        # Configure the ViT encoder
        config = ViTConfig(
            image_size=224,           # Input image size
            patch_size=16,            # Size of image patches
            num_channels=3,           # RGB channels
            hidden_size=768,          # Embedding size
            num_attention_heads=12,   # Number of attention heads
            num_hidden_layers=6       # Number of transformer blocks
        )

        # Initialize the Vision Transformer encoder
        self.encoder = ViTModel(config)

        # Define a decoder that gradually upsamples feature maps to output a 2D mask
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # Upsample again
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),                        # Final 1-channel output
            nn.Sigmoid()                                            # Normalize output to [0, 1]
        )

    def forward(self, x):
        # Pass input image through the ViT encoder
        x = self.encoder(pixel_values=x).last_hidden_state  # Output: (batch_size, num_patches+1, hidden_size)

        # Extract shape parameters
        b, n, c = x.shape

        # Remove the class token and reshape to 2D feature map
        x = x[:, 1:, :].permute(0, 2, 1).reshape(b, c, 14, 14)  # Convert to (batch, channels, height, width)

        # Pass through the decoder to get the predicted segmentation mask
        x = self.decoder(x)

        # Upsample the output to match original input size (224x224)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        return x


# Function to load the trained model from saved weights
def load_model(path):
    model = ViTUNet()  # Initialize model
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))  # Load weights
    model.eval()  # Set to evaluation mode
    return model
def predict_mask(model, image):
    # Define preprocessing: resize and convert image to tensor
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization values
    ])

    pil_image = Image.fromarray(image)

    # Apply transform and add batch dimension
    image_tensor = transform(pil_image).unsqueeze(0)

    # Disable gradient tracking for inference
    with torch.no_grad():
        output = model(image_tensor)  # Get first image in batch and squeeze output

    # Apply threshold to convert probabilities to binary mask
    mask = (output > 0.5).float().numpy()

    # Squeeze the unnecessary dimensions (remove batch and channel)
    mask = mask.squeeze()  # Resulting shape should be (224, 224)

    # Convert mask to PIL Image format for display
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    return mask_img


def vit_based(input_image):
    file_id = '1FbZ77_AmIy0_DFM1g5XC_0wZDEJlCFTS'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'

    # Download the file
    gdown.download(url, 'vit_teeth_segmentation.pth', quiet=False)
    model = load_model("vit_teeth_segmentation.pth")
    mask = predict_mask(model, input_image)
    col3.image(
                mask,
                caption="Predicted Result Using ViT-UNet",
                use_column_width=True,
            )
    return mask

def image_processing(input_image):
    print(input_image.shape)
    if len(input_image.shape) == 3 and input_image.shape[2] == 3:
    # If the image has 3 channels (BGR), convert it to grayscale
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
    # If the image is already grayscale (1 channel), use it as is
        gray_image = input_image
    resized_image = cv2.resize(gray_image, (256, 256))
    

    blurred = cv2.GaussianBlur(resized_image, (5,5), 0)
    equalized = cv2.equalizeHist(blurred)

    # 3. Global Threshold (Otsu's)
    _, global_thresh = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Adaptive Threshold
    block_size = 15  # try 11, 15, 21, etc.
    C = 5            # try 2, 5, 7...
    adaptive_thresh = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    # 5. Combine the Two Masks
    #    A) Intersection (logical AND): keeps only pixels white in both
    combined_and = cv2.bitwise_and(global_thresh, adaptive_thresh)

    #    B) Union (logical OR): keeps pixels white if either threshold is white
    combined_or = cv2.bitwise_or(global_thresh, adaptive_thresh)

    # Depending on your data, pick AND or OR or some custom logic.
    # If you want to remove "extra" noise, try AND. 
    # If you want to fill in missing parts, try OR.

    # Let's assume we want the intersection to remove extra elements
    combined_mask = combined_and

    # 6. Morphological Cleanup
    kernel = np.ones((3,3), np.uint8)
    # - Closing to fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # - Opening to remove small noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 7. Optional: Remove Tiny or Large Extraneous Regions
    num_labels, labels = cv2.connectedComponents(combined_mask)
    min_area = 50  # adjust to remove small noise
    for label_id in range(1, num_labels):
        area = np.sum(labels == label_id)
        if area < min_area:
            combined_mask[labels == label_id] = 0



    # offset = 2  # try 5, 10, 20, etc. to see what works best
    # h, w = combined_mask.shape
    # floodfill_mask = np.zeros((h+2, w+2), np.uint8)

    # # # # Top (offset) and bottom (h-1-offset) rows
    # # for x in range(offset, w - offset):
    # #     # flood fill from row offset
    # #     if combined_mask[offset, x] == 255:
    # #         cv2.floodFill(combined_mask, floodfill_mask, (x, offset), 0)
    # #     # flood fill from row h-1-offset
    # #     if combined_mask[h-1-offset, x] == 255:
    # #         cv2.floodFill(combined_mask, floodfill_mask, (x, h-1-offset), 0)

    # # Left (offset) and right (w-1-offset) columns
    # for y in range(offset, h - offset):
    #     # flood fill from column offset
    #     if combined_mask[y, offset] == 255:
    #         cv2.floodFill(combined_mask, floodfill_mask, (offset, y), 0)
    #     # flood fill from column w-1-offset
    #     if combined_mask[y, w-1-offset] == 255:
    #         cv2.floodFill(combined_mask, floodfill_mask, (w-1-offset, y), 0)
    
    backtorgb = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)
    backtorgb[np.where((backtorgb == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  #bgr

    num_labels, labels_im = cv2.connectedComponents(combined_mask)
 

    # #
    label_hue = np.uint8(179 * labels_im / np.max(labels_im))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cv2.imwrite('l1.jpg',labeled_img)

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    # cv2.imwrite('l2.jpg', labeled_img)
    # color_mask_hsv = cv2.cvtColor(backtorgb, cv2.COLOR_BGR2HSV)
    color_mask_hsv = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2HSV)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
    img_hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.6

    img_masked = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    final = cv2.resize(img_masked,(256,256))
    col2.image(
                final,
                caption="Predicted Result Using Image Processing",
                use_column_width=True,
            )



    return final


def predict(resized_image):
    # Load the model only once at the start, outside the predict function
    import requests
    file_id = '1d5Yka9qb4Rd22PWxsGgfjOnx_wNJssLQ'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'

    # Download the file
    gdown.download(url, 'teethsegmentation_34.hdf5', quiet=False)
    model = keras.models.load_model('teethsegmentation_34.hdf5', compile=False)


    # Check if the image is grayscale and convert if necessary
    if len(resized_image.shape) == 2:  # Grayscale image (256, 256)
        resized_image = np.expand_dims(resized_image, axis=-1)  # Convert shape to (256, 256, 1)

    # Ensure that the image is in grayscale format (1 channel)
    if resized_image.shape[-1] == 3:  # If the image is RGB, convert it to grayscale
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized_image = np.expand_dims(resized_image, axis=-1)  # Convert shape to (256, 256, 1)

    # Normalize the image
    normalized_image = resized_image.astype('float32') / 255.0  # Normalize to [0, 1]

    # Expand dimensions to match the model's input shape (batch size, height, width, channels)
    input_image = np.expand_dims(normalized_image, axis=0)  # Shape becomes (1, 256, 256, 1)

    # Make the prediction
    pred_mask = model.predict(input_image)

    # Convert the probabilities to a binary mask using a threshold (e.g., 0.5)
    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # Convert the predicted binary mask back to a format suitable for display
    pred_mask_image = cv2.cvtColor(binary_mask[0], cv2.COLOR_GRAY2RGB)  # Convert to RGB for display

    # Convert the probabilities to a binary mask using a threshold (e.g., 0.5)
    binary_mask = (pred_mask_image > 0.5).astype(np.uint8)

    print("binaryimage",binary_mask.shape)
    prediction_image = binary_mask
    # binary_mask.reshape(resized_image.shape)
    #print("PI:", prediction_image.shape)
    prediction_image_8 = (prediction_image * 255).astype(np.uint8)
    print(prediction_image_8.shape)


    img1 = cv2.threshold(prediction_image_8, 127, 255, cv2.THRESH_BINARY)[1]
    backtorgb = img1
    # backtorgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    backtorgb[np.where((backtorgb == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  #bgr
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    num_labels, labels_im = cv2.connectedComponents(img1)
 

    # #
    label_hue = np.uint8(179 * labels_im / np.max(labels_im))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    cv2.imwrite('l1.jpg',labeled_img)

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    # cv2.imwrite('l2.jpg', labeled_img)
    # color_mask_hsv = cv2.cvtColor(backtorgb, cv2.COLOR_BGR2HSV)
    
    color_mask_hsv = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2HSV)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
    img_hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.6

    img_masked = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    final = cv2.resize(img_masked,(256,256))
    col2.image(
                final,
                caption="Predicted Result Using Deep Neural Network",
                use_column_width=True,
            )

    return final


def main():
    st.sidebar.title("Dental Vision: Automated Teeth Segmentation for Dental Diagnostics")

    # Add a description for the app
    st.sidebar.markdown(
        "Dental Vision Highlighter: refers to the use of advanced computer vision and deep learning techniques to automatically detect, segment, and analyze teeth structures in dental images. This automated process assists in diagnosing dental conditions, planning treatments, and monitoring oral health by accurately identifying and segmenting teeth from X-rays, CT scans, or intraoral images. The goal of DentalVision is to streamline the workflow for dental professionals, reduce human error, and enhance the precision of diagnostics, leading to better patient outcomes. The system utilizes state-of-the-art algorithms like convolutional neural networks (CNNs) to achieve high accuracy in segmentation and analysis."  )

    # Create an "Upload File" button
    uploaded_files = st.sidebar.file_uploader(
        "***Upload JPG or PNG image***", type=["jpg", "png"], accept_multiple_files=True
    )

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_name = os.path.splitext(file_name)[0]

            # Convert the uploaded image to a PIL image
            pil_image = Image.open(uploaded_file)
            img = np.asarray(pil_image)
            resized_image = cv2.resize(img, (256, 256))
            # main_img = cv2.resize(img, (640, 480))
            # col1, col2, col3 = st.columns(3)
            col1.image(resized_image, caption="Uploaded Image", use_column_width=True)

            # Add a progress bar while the image is being processed
            progress_text = "Processing the image..."
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.05)  # Simulating a small delay for processing
                progress_bar.progress(percent_complete + 1)

            # Perform the prediction on the uploaded image
            ip_output_image = image_processing(img)
            ip_output_image = cv2.resize(ip_output_image, (256, 256))

            output_image = predict(resized_image)

            output_image = cv2.resize(output_image, (256, 256))
            # Use st.beta_columns to display images side by side

            # Display the predicted result in the second column
            
            output_vit = vit_based(output_image)
            # col2.image(
            #     ip_output_image,
            #     caption="Predicted Result Using Image Processing",
            #     use_column_width=True,
            # )

            # col2.image(
            #     output_image,
            #     caption="Predicted Result Using Deep Neural Network",
            #     use_column_width=True,
            # )

            # col3.image(
            #     output_vit,
            #     caption="Predicted Result Using ViT-UNet",
            #     use_column_width=True,
            # )

            # concatenated_image = Image.fromarray(
            #     np.concatenate(
            #         [
            #             np.array(resized_image),
            #             np.array(ip_output_image),
            #             np.array(output_image),
            #         ],
            #         axis=1,
            #     )
            # )
            # concatenated_image_bytes = io.BytesIO()
            # concatenated_image.save(concatenated_image_bytes, format="JPEG")
            # concatenated_image_bytes.seek(0)

            # # Download concatenated image
            # st.download_button(
            #     "Download Concatenated Image",
            #     data=concatenated_image_bytes,
            #     file_name=f"concatenated_image_{file_name}.jpg",
            # )

            # Clear the progress bar after displaying the predicted result
            progress_bar.empty()


if __name__ == "__main__":
    main()
