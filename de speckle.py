import numpy as np
import pywt
from astropy.io import fits
import matplotlib.pyplot as plt
import os

def get_wavelet_choice():
    wavelet_options = ['bior6.8', 'coif3', 'db4', 'haar', 'sym8']
    print("\nAvailable wavelet options:", wavelet_options)
    wavelet = input("Choose a wavelet (default: 'bior6.8'): ").strip()
    return wavelet if wavelet in wavelet_options else 'bior6.8'

def wavelet_denoise(image, wavelet='bior6.8', level=3, threshold_factor=1.0):
    print("\n🔄 Applying wavelet despeckling...")
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs_thresh = [coeffs[0]]
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        threshold_H = np.median(np.abs(cH)) * threshold_factor
        threshold_V = np.median(np.abs(cV)) * threshold_factor
        threshold_D = np.median(np.abs(cD)) * threshold_factor
        cH = pywt.threshold(cH, threshold_H, mode='soft')
        cV = pywt.threshold(cV, threshold_V, mode='soft')
        cD = pywt.threshold(cD, threshold_D, mode='soft')
        coeffs_thresh.append((cH, cV, cD))
    return pywt.waverec2(coeffs_thresh, wavelet)

def apply_threshold(image, thresholds, method_name, wavelet_type):
    print("\n📊 Applying thresholding to despeckled image...")
    fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 5), constrained_layout=True)
    for ax, threshold in zip(axes, thresholds):
        modified_image = np.copy(image)
        modified_image[modified_image <= threshold] = np.nan
        im = ax.imshow(modified_image, origin='lower', cmap='gray')
        ax.set_title(f'Threshold > {threshold}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
    fig.suptitle(f'Method: {method_name} (Wavelet: {wavelet_type})\nPhysical Properties: Intensity Distribution', fontsize=14)
    plt.show()

def main():
    fits_file = "C:\\Users\\ASUS\\Desktop\\cropped_DDO 154_H.fits"
    wavelet_type = get_wavelet_choice()
    threshold_levels = [2, 4, 6, 8, 10]
    
    print("\n📂 Loading FITS file...")
    with fits.open(fits_file) as hdul:
        if len(hdul) == 0 or hdul[0].data is None:
            print("❌ Error: FITS file does not contain image data!")
            exit(1)
        
        image = hdul[0].data
    
    if len(image.shape) > 2:
        print("⚠️ Warning: FITS image has multiple dimensions. Using the first 2D slice.")
        image = image[0]
    image = image.astype(np.float32)
    
    denoised_image = wavelet_denoise(image, wavelet=wavelet_type, threshold_factor=1.0)
    apply_threshold(denoised_image, threshold_levels, "Wavelet Despeckling", wavelet_type)
    print("\n✅ Processing complete! You can now analyze your images.")

if __name__ == "__main__":
    main()
