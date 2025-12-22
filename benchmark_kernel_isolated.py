
import time
import numpy as np
from scipy import integrate

def cubic(r):
    """Calculate the cubic spline kernel.

    Args:
        r (float): The distance from the center of the kernel.

    Returns:
        float: The value of the cubic spline kernel.
    """
    if r < 0.5:
        return 8 / np.pi * (1 - 6 * r**2 + 6 * r**3)
    elif r < 1.0:
        return 16 / np.pi * (1 - r) ** 3
    else:
        return 0.0

class Kernel:
    def __init__(self, name="cubic", binsize=10000):
        self.name = name
        self.binsize = binsize
        if name == "cubic":
            self.f = cubic
        else:
            raise ValueError("Kernel name not defined")

    def W_dz(self, z, b):
        return self.f(np.sqrt(z**2 + b**2))

    def _integral_func(self, ii):
        return lambda z: self.W_dz(z, ii)

    def get_kernel(self):
        kernel = np.zeros(self.binsize + 1)
        bins = np.arange(0, 1.0, 1.0 / self.binsize)
        bins = np.append(bins, 1.0)

        for ii in range(self.binsize):
            y, yerr = integrate.quad(
                self._integral_func(bins[ii]), 0, np.sqrt(1.0 - bins[ii] ** 2)
            )
            kernel[ii] = y * 2.0

        return kernel

print("Starting kernel benchmark (isolated)...")
start = time.time()
kernel_obj = Kernel(name="cubic")
kernel = kernel_obj.get_kernel()
end = time.time()
print(f"Kernel calculation took {end - start:.2f} seconds")
