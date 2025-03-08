import numpy as np
import pywt
from astropy.io import fits
import matplotlib.pyplot as plt

def get_wavelet_choice():
    wavelet_options = ['bior6.8', 'coif3', 'db4', 'haar', 'sym8']
    print("\nAvailable wavelet options:", wavelet_options)
    wavelet = input("Choose a wavelet (default: 'bior6.8'): ").strip()
    return wavelet if wavelet in wavelet_options else 'bior6.8'

def get_wavelet_scale():
    """Ask user for wavelet decomposition scale."""
    try:
        level = int(input("Enter maximum wavelet decomposition scale (default: 4): ").strip())
        if level < 1 or level > 10:  # Keep it in a reasonable range
            raise ValueError
        return level
    except ValueError:
        print("⚠️ Invalid input! Using default scale: 4")
        return 4

def get_plot_range(level):
    """Ask user for the range of wavelet scales to plot."""
    try:
        min_scale = int(input(f"Enter minimum scale to plot (1-{level}): ").strip())
        max_scale = int(input(f"Enter maximum scale to plot (1-{level}): ").strip())
        
        if min_scale < 1 or max_scale > level or min_scale > max_scale:
            raise ValueError
        return min_scale, max_scale
    except ValueError:
        print(f"⚠️ Invalid input! Using full range (1-{level}).")
        return 1, level

def compute_wavelet_energy(image, wavelet='bior6.8', level=4):
    """Computes energy at each wavelet scale."""
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    energy_levels = []
    for i, coeff in enumerate(coeffs[1:], 1):  # Skip approximation coefficients
        cH, cV, cD = coeff
        energy = np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)
        energy_levels.append(energy)
    
    return energy_levels

def plot_energy_levels(energy_levels, wavelet_type, min_scale, max_scale):
    """Plots energy levels within user-defined range."""
    plt.figure(figsize=(8, 5))
    
    scales = np.arange(1, len(energy_levels) + 1)
    selected_scales = scales[(scales >= min_scale) & (scales <= max_scale)]
    selected_energies = energy_levels[min_scale - 1:max_scale]

    plt.plot(selected_scales, selected_energies, marker='o', linestyle='-', color='b')
    plt.xlabel("Wavelet Scale (Level)")
    plt.ylabel("Energy")
    plt.title(f"Energy Distribution Across Wavelet Scales ({wavelet_type})\nPlotted Range: {min_scale}-{max_scale}")
    plt.grid(True)
    plt.show()

def main():
    fits_file = "C:\\Users\\ASUS\\Desktop\\un cleaned codes\\hbeta-2023_11_09-exp00.03.00.000-4x4_High_3.fit"
    wavelet_type = get_wavelet_choice()
    wavelet_level = get_wavelet_scale()
    min_scale, max_scale = get_plot_range(wavelet_level)

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
    
    # Compute wavelet energy
    energy_levels = compute_wavelet_energy(image, wavelet=wavelet_type, level=wavelet_level)
    
    # Plot energy distribution within user-defined range
    plot_energy_levels(energy_levels, wavelet_type, min_scale, max_scale)

    print("\n✅ Energy computation complete!")

if __name__ == "__main__":
    main()
