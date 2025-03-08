import numpy as np
import pywt
from astropy.io import fits
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter

def get_wavelet_choice():
    wavelet_options = ['bior6.8', 'coif3', 'db4', 'haar', 'sym8']
    print("\nAvailable wavelet options:", wavelet_options)
    wavelet = input("Choose a wavelet (default: 'bior6.8'): ").strip()
    return wavelet if wavelet in wavelet_options else 'bior6.8'

def get_stretch_choice():
    stretch_options = ['linear', 'log', 'sqrt', 'autostretch']
    print("\nAvailable image stretching methods:", stretch_options)
    stretch = input("Choose a stretch method (default: 'linear'): ").strip().lower()
    return stretch if stretch in stretch_options else 'linear'

def apply_stretching(image, stretch_type):
    image = np.nan_to_num(image, nan=0)
    image -= np.min(image)
    
    if np.max(image) > 0:
        image /= np.max(image)
    
    if stretch_type == 'log':
        return np.log1p(image)
    elif stretch_type == 'sqrt':
        return np.sqrt(image)
    elif stretch_type == 'autostretch':
        p_low, p_high = np.percentile(image, (0.5, 99.5))
        return np.clip((image - p_low) / (p_high - p_low), 0, 1)
    
    return image

def remove_hot_pixels(image, threshold_sigma=5):
    """Detect and correct hot pixels using sigma clipping and median filtering."""
    print("\n🛠 Removing hot pixels...")
    median_filtered = median_filter(image, size=3)
    diff = np.abs(image - median_filtered)
    
    std_dev = np.std(diff)
    mask = diff > (threshold_sigma * std_dev)  # Detect extreme outliers
    
    corrected_image = np.copy(image)
    corrected_image[mask] = median_filtered[mask]  # Replace hot pixels
    
    print(f"✔️ Removed {np.sum(mask)} hot pixels.")
    return corrected_image

def wavelet_denoise(image, wavelet='bior6.8', threshold_factor=1.5, mode='hard'):
    print("\n🔄 Applying wavelet despeckling...")
    
    max_level = pywt.dwt_max_level(image.shape[0], pywt.Wavelet(wavelet).dec_len)
    level = min(4, max_level)
    
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs_thresh = [coeffs[0]]
    
    sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745
    universal_threshold = sigma * np.sqrt(2 * np.log(image.size)) * threshold_factor
    
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        cH = pywt.threshold(cH, universal_threshold, mode=mode)
        cV = pywt.threshold(cV, universal_threshold, mode=mode)
        cD = pywt.threshold(cD, universal_threshold, mode=mode)
        coeffs_thresh.append((cH, cV, cD))
    
    return pywt.waverec2(coeffs_thresh, wavelet)

def apply_threshold(image, thresholds, method_name, wavelet_type, stretch_type):
    print("\n📊 Applying thresholding with stretch:", stretch_type)
    
    fig, axes = plt.subplots(1, len(thresholds) + 1, figsize=(18, 5), constrained_layout=True)
    
    # Plot the original denoised image
    stretched_main = apply_stretching(image, stretch_type)
    im = axes[0].imshow(stretched_main, origin='lower', cmap='gray', interpolation='nearest')
    axes[0].set_title('Denoised Image')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='Intensity')

    # Plot thresholded images
    for ax, threshold in zip(axes[1:], thresholds):
        modified_image = np.copy(image)
        modified_image[modified_image <= threshold] = np.nan
        
        stretched_image = apply_stretching(modified_image, stretch_type)
        im = ax.imshow(stretched_image, origin='lower', cmap='gray', interpolation='nearest')
        
        ax.set_title(f'Threshold > {threshold}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
    
    fig.suptitle(f'Method: {method_name} (Wavelet: {wavelet_type})\nStretch: {stretch_type}\nPhysical Properties: Intensity Distribution', fontsize=14)
    plt.show()

def main():
    fits_file = "C:\\Users\\ASUS\\Desktop\\un cleaned codes\\hbeta-2023_11_09-exp00.03.00.000-4x4_High_3.fit"
    wavelet_type = get_wavelet_choice()
    stretch_type = get_stretch_choice()
    threshold_levels = [40, 60, 80, 100] 
    
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
    
    cleaned_image = remove_hot_pixels(image)
    denoised_image = wavelet_denoise(cleaned_image, wavelet=wavelet_type, threshold_factor=1.5, mode='hard')
    apply_threshold(denoised_image, threshold_levels, "Wavelet Despeckling", wavelet_type, stretch_type)
    
    print("\n✅ Processing complete! You can now analyze your images.")

if __name__ == "__main__":
    main()

