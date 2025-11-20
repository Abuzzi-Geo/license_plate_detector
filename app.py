import streamlit as st
import cv2
import numpy as np
import easyocr
import imutils
from PIL import Image

st.title("OCR Application with Streamlit")

# Function to perform OCR
def perform_ocr(image):
    # Convert image to OpenCV format
    img_np = np.array(image.convert('RGB'))
    img_cv2 = img_np[:, :, ::-1].copy() # Convert RGB to BGR for OpenCV

    # Grayscale and bilateral filter
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # Canny edge detection
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        return None, "No rectangular contour found.", None

    # Create mask and new image
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img_cv2, img_cv2, mask=mask)

    # Crop image based on mask
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # EasyOCR reader
    reader = easyocr.Reader(['en']) # Initialize only once if possible in a real app
    result = reader.readtext(cropped_image)

    detected_text = ""
    if result:
        detected_text = result[0][-2]

        # Draw text and rectangle on the original image (cv2 format)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Adjust text position to be relative to the original image's coordinates
        # location is already in the context of the original image
        text_pos_x = location[0][0][0] # x-coordinate of the top-left corner of the contour
        text_pos_y = location[1][0][1] + 60 # y-coordinate slightly below the contour

        res_img = img_cv2.copy() # Make a copy to draw on
        res_img = cv2.putText(res_img, text=detected_text, org=(text_pos_x, text_pos_y), 
        fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        res_img = cv2.rectangle(res_img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
    else:
        res_img = img_cv2.copy() # If no text, just use the original image

    return img_cv2, cropped_image, detected_text, res_img

# Streamlit File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    st.spinner("Processing image and detecting text...")
    with st.spinner('Performing OCR...'):
        original_cv2_img, cropped_image, detected_text, final_image_with_text = perform_ocr(image)

    if original_cv2_img is not None:
        st.subheader("Detected Text:")
        st.write(f"**{detected_text}**")

        st.subheader("Cropped Region for OCR:")
        st.image(cropped_image, caption='Cropped Image', use_column_width=True, channels='GRAY')

        st.subheader("Final Image with Detected Text and Bounding Box:")
        st.image(final_image_with_text, caption='Result', use_column_width=True, channels='BGR')
    else:
        st.error(detected_text)