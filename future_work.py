import streamlit as st
import cv2
import numpy as np

# Function to calculate pixel-to-meter conversion factor
def pixel_to_meter_factor(image_width_in_pixels, image_width_in_meters):
    return image_width_in_meters / image_width_in_pixels

# Title of your Streamlit app
st.title('Particle Segmentation App')

# Apply custom CSS to change the background color to black
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# File uploader widget to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    # resizing
    image = cv2.resize(image, (612, 433))

    # Specify the physical size of the image (in meters)
    image_width_in_meters = 5.0  # Example value, replace with your actual image size

    # Calculate pixel-to-meter conversion factor
    conversion_factor = pixel_to_meter_factor(image.shape[1], image_width_in_meters)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations to clean the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Find sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to markers to avoid labeling the background as 0
    markers = markers + 1

    # Mark unknown region as 0 in markers
    markers[unknown == 255] = 0

    # Apply watershed algorithm
    cv2.watershed(image, markers)
    markers[markers == -1] = 0  # Remove watershed lines

    # Count the number of particles (excluding background)
    num_particles = np.max(markers) - 1

    # Display the number of particles in the center, bold, and with a larger font size
    st.markdown(f"<p style='text-align:center;font-size:24px;font-weight:bold;'>Number of particles: {num_particles}</p>", unsafe_allow_html=True)

    # Calculate and display the radius of each particle in meters
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        # Calculate the equivalent radius assuming the contour is a circle
        radius_pixels = np.sqrt(cv2.contourArea(contour) / np.pi)
        radius_meters = radius_pixels * conversion_factor
        st.write(f"Radius of particle {i + 1}: {radius_meters:.4f} meters")

    # Draw boundaries on the original image
    image[markers > 1] = [0, 0, 255]  # Mark boundaries in red

    # Display the result
    st.image(image, channels="BGR")
