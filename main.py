import os
import numpy as np
from pydicom import dcmread
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle # Added to rectangle patch to a plot

#### Parts of the code is done with cooperation with ChatGPT: 
#### Bounding boxes; lines 70-106
#### Also to understand Pixel Shading and pydicom 

# Function to read coordinates from the .pts file
def coordinates(points_file):
    line_count = 0
    x = []
    y = []
    with open(points_file) as fp:
        for line in fp:
            if line_count >= 1:
                end_point = 151
            if 3 <= line_count < end_point:
                x_temp = line.split(" ")[0]
                y_temp = line.split(" ")[1]
                y_temp = y_temp.replace("\n", "")
                x.append((float(x_temp)))
                y.append((float(y_temp)))
            line_count += 1
    return x, y

dicom_dir = 'data/dicoms'
landmarks_dir = 'data/landmarks'
output_dir = 'data/cropped_images'  # Directory to save the cropped images

# Ensuring that the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for dcm in os.listdir(dicom_dir):
    dcm_name = os.path.splitext(dcm)[0]

    dcm_path = os.path.join(dicom_dir, dcm_name)
    dicom_data = dcmread(dcm_path)

    landmark_path = os.path.join(landmarks_dir, f"{dcm_name}.pts")
    Xs, Ys = coordinates(landmark_path)

    ############## Showing the original image ###############
    plt.imshow(dicom_data.pixel_array, cmap=plt.cm.gray)
    plt.title(f"Patient ID: {dicom_data.PatientID}")

    # Get pixel spacing in mm/pixel, it is typically an array: [row_spacing, column_spacing]
    pixel_spacing = dicom_data.PixelSpacing  
    pixel_spacing_mm = np.array(pixel_spacing, dtype=float)

    # Convert 1 cm to pixels
    margin_cm = 10  # 1 cm in millimeters
    margin_pixels = margin_cm / pixel_spacing_mm

    # Split the landmarks into left and right knee regions
    mid_x = np.median(Xs)  # Midpoint to separate left and right knee landmarks

    left_knee_Xs = [x for x in Xs if x < mid_x]
    left_knee_Ys = [Ys[i] for i, x in enumerate(Xs) if x < mid_x]

    right_knee_Xs = [x for x in Xs if x >= mid_x]
    right_knee_Ys = [Ys[i] for i, x in enumerate(Xs) if x >= mid_x]

    ######### Bounding boxes for both knees is done separately #########

    # Get bounding boxes for both knees
    def get_bounding_box(Xs, Ys, margin_pixels, dicom_data):
        min_x, max_x = min(Xs), max(Xs)
        min_y, max_y = min(Ys), max(Ys)
        start_x = max(0, int(min_x - margin_pixels[1]))  # columns spacing
        end_x = min(dicom_data.pixel_array.shape[1], int(max_x + margin_pixels[1]))
        start_y = max(0, int(min_y - margin_pixels[0]))  # row spacing
        end_y = min(dicom_data.pixel_array.shape[0], int(max_y + margin_pixels[0]))
        return start_x, end_x, start_y, end_y

    # Bounding box for left knee
    start_x_left, end_x_left, start_y_left, end_y_left = get_bounding_box(
        left_knee_Xs, left_knee_Ys, margin_pixels, dicom_data)

    # Bounding box for right knee
    start_x_right, end_x_right, start_y_right, end_y_right = get_bounding_box(
        right_knee_Xs, right_knee_Ys, margin_pixels, dicom_data)

    ############# Cropping and saving the images #############
    cropped_left_knee = dicom_data.pixel_array[start_y_left:end_y_left, start_x_left:end_x_left]
    cropped_right_knee = dicom_data.pixel_array[start_y_right:end_y_right, start_x_right:end_x_right]

    output_filename_left = os.path.join(output_dir, f"{dicom_data.PatientID}_L.png")
    output_filename_right = os.path.join(output_dir, f"{dicom_data.PatientID}_R.png")
    plt.imsave(output_filename_left, cropped_left_knee, cmap='gray')
    plt.imsave(output_filename_right, cropped_right_knee, cmap='gray')
    print(f"Saved cropped images: {output_filename_left}, {output_filename_right}")

    ############# Drawing the bounding boxes #############
    # Left knee bounding box
    rect_left = Rectangle((start_x_left, start_y_left), end_x_left - start_x_left, end_y_left - start_y_left, 
                          linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect_left)

    # Right knee bounding box
    rect_right = Rectangle((start_x_right, start_y_right), end_x_right - start_x_right, end_y_right - start_y_right, 
                           linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect_right)

    ############## Scatter landmarks for both knees ###############
    plt.scatter(Xs, Ys, c='green', marker='.', s=10)

    ############## Annotate the 1 cm margin around both boxes ###############
    plt.text((start_x_left + end_x_left) / 2, start_y_left - 10, "1cm", color='pink', ha='center', fontsize=12)
    plt.text(start_x_left - 10, (start_y_left + end_y_left) / 2, "1cm", color='pink', va='center', rotation=90, fontsize=12)

    plt.text((start_x_right + end_x_right) / 2, start_y_right - 10, "1cm", color='pink', ha='center', fontsize=12)
    plt.text(start_x_right - 10, (start_y_right + end_y_right) / 2, "1cm", color='pink', va='center', rotation=90, fontsize=12)

    ############# Show the final image with the bounding boxes #############
    plt.title(f"Patient ID: {dicom_data.PatientID} with 1 cm margin around each knee")
    plt.show()

    print(f"Bounding boxes drawn and cropped images saved for both knees (Patient ID: {dicom_data.PatientID})")


####### Emil Eskolin #######
### As you can see I did a lot of modifications to the original code
### Quite possible that this code is not optimal, I would argue that it is too complicated to achieve the result...