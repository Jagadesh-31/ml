import numpy as np
from scipy.datasets import ascent
import matplotlib.pyplot as plt

# Load image and convert to wider int type
i = ascent().astype(np.int16)
i_transformed = np.copy(i).astype(float)

# Sobel horizontal edge detection filter
filter = [ [-1, 0, 1], [0, 0, 0], [-2, 0 ,2] ]
weight = 1

for x in range(1, i.shape[0] - 1):
    for y in range(1, i.shape[1] - 1):
        acc = 0.0
        for fx in range(3):
            for fy in range(3):
                acc += i[x + fx - 1, y + fy - 1] * filter[fx][fy]
        acc *= weight
        acc = max(0, min(255, acc))  # Clamp
        i_transformed[x, y] = acc

# Display filtered image
plt.imshow(i_transformed, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()

# Max pooling / downsampling
new_x, new_y = i.shape[0] // 2, i.shape[1] // 2
newImage = np.zeros((new_x, new_y))

for x in range(0, i.shape[0] - 1, 2):
    for y in range(0, i.shape[1] - 1, 2):
        patch = [
            i_transformed[x, y],
            i_transformed[x+1, y],
            i_transformed[x, y+1],
            i_transformed[x+1, y+1]
        ]
        newImage[x//2, y//2] = max(patch)

plt.imshow(newImage, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()
