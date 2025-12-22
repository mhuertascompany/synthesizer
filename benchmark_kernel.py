
import time
from synthesizer.kernel_functions import Kernel

print("Starting kernel benchmark...")
start = time.time()
kernel_obj = Kernel(name="cubic")
kernel = kernel_obj.get_kernel()
end = time.time()
print(f"Kernel calculation took {end - start:.2f} seconds")
