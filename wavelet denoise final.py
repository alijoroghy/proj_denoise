import numpy as np
import pywt
from astropy.io import fits
import matplotlib.pyplot as plt
import os

# ---------------------- User Input Functions ----------------------

def get_fits_path():
    """Prompt user for a FITS file path."""
    fits_path = input("Enter the full path to your FITS file: ").strip()
    while not os.path.exists(fits_path):
        print("⚠️ File not found! Please enter a valid path.")
        fits_path = input("Enter the full path to your FITS file: ").strip()
    return fits_path

def get_wavelet_choice():
    """Prompt user to select a wavelet type."""
    wavelet_options = ['bior6.8', 'coif3', 'db4', 'haar', 'sym8']
    print("\nAvailable wavelet options:", wavelet_options)
    wavelet = input("Choose a wavelet (default: 'bior6.8'): ").strip()
    return wavelet if wavelet in wavelet_options else 'bior6.8'

def get_threshold_levels():
    """Prompt user to enter threshold levels"""
    try:
        levels = input("Enter threshold levels separated by commas (default: 2, 4, 6, 8, 10): ").strip()
        return [float(x) for x in levels.split(",")] if levels else [2, 4, 6, 8, 10]
    except ValueError:
        print("⚠️ Invalid input! Using default threshold levels.")
        return [2, 4, 6, 8, 10]

def get_denoise_strength():
    """Prompt user to set denoising strength."""
    try:
        strength = float(input("Enter denoising strength (default: 1.0, lower = less smoothing, higher = more smoothing): ").strip())
        return max(0.1, min(2.0, strength))  # Keep within reasonable bounds
    except ValueError:
        print("⚠️ Invalid input! Using default strength: 1.0")
        return 1.0

# ---------------------- Wavelet Denoising Function ----------------------

def wavelet_denoise(image, wavelet='bior6.8', level=3, threshold_factor=1.0):
    """Denoise an image using adaptive wavelet shrinkage while preserving details."""
    print("\n🔄 Applying wavelet despeckling...")

    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients unchanged

    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]

        # Adaptive threshold: Higher for noisy areas, lower for important details
        threshold_H = np.median(np.abs(cH)) * threshold_factor
        threshold_V = np.median(np.abs(cV)) * threshold_factor
        threshold_D = np.median(np.abs(cD)) * threshold_factor

        cH = pywt.threshold(cH, threshold_H, mode='soft')
        cV = pywt.threshold(cV, threshold_V, mode='soft')
        cD = pywt.threshold(cD, threshold_D, mode='soft')

        coeffs_thresh.append((cH, cV, cD))

    return pywt.waverec2(coeffs_thresh, wavelet)

# ---------------------- Thresholding Function ----------------------

def apply_threshold(image, thresholds, filter_type):
    """Apply thresholding to an image and display results."""
    print("\n📊 Applying thresholding to despeckled image...")
    
    for threshold in thresholds:
        modified_image = np.copy(image)
        modified_image[modified_image <= threshold] = np.nan  # Mask pixels below threshold

        plt.figure(figsize=(8, 8))
        plt.imshow(modified_image, origin='lower', cmap='gray')
        plt.title(f'Threshold > {threshold} for {filter_type}')
        plt.colorbar()
        plt.show()

        # Calculate the sum of non-NaN values
        sum_values = np.nansum(modified_image)
        print(f'✔️ Sum of pixel values above threshold {threshold}: {sum_values:.2f}\n')

# ---------------------- Main Execution ----------------------

# Get user inputs
fits_file = get_fits_path()
wavelet_type = get_wavelet_choice()
threshold_levels = get_threshold_levels()
denoise_strength = get_denoise_strength()

# Load FITS image
print("\n📂 Loading FITS file...")
with fits.open(fits_file) as hdul:
    image = hdul[0].data

# Ensure image is 2D (grayscale)
if len(image.shape) > 2:
    print("⚠️ Warning: FITS image has multiple dimensions. Using the first 2D slice.")
    image = image[0]  # Take the first slice if it's a 3D cube

image = image.astype(np.float32)  # Convert to float for processing

# Apply wavelet despeckling
denoised_image = wavelet_denoise(image, wavelet=wavelet_type, threshold_factor=denoise_strength)

# Apply thresholding
apply_threshold(denoised_image, threshold_levels, "Wavelet Despeckled")

print("\n✅ Processing complete! You can now analyze your images.")
