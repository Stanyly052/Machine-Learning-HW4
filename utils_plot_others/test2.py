import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import fftpack

im = Image.open("/home/yuzhi/ML_HW_2/cat.png").convert("L")

im_fft = fftpack.fft2(im)

# Show the results

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.real(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

plt.figure()
plot_spectrum(im_fft)
plt.title('2D Discrete Fast Fourier transform')
plt.savefig("StreetView_real.png")