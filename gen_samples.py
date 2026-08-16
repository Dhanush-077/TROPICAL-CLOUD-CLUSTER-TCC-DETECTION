import os
import numpy as np
import cv2

os.makedirs('sample_tcc_images', exist_ok=True)

def generate_blob(size=64, center=(32,32), sigma=10, min_val=50):
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    # Background is warm (e.g. 200) representing clear sky/ocean
    # Center is cold (e.g. min_val) representing high cloud tops
    img = 200 - (200 - min_val) * np.exp(- (dist**2) / (2.0 * sigma**2))
    return img.astype(np.uint8)

# 1. Cyclone (large, very cold core < 50)
img1 = generate_blob(size=64, center=(32,32), sigma=12, min_val=30)
# Add realistic noise
img1 = np.clip(img1 + np.random.normal(0, 5, img1.shape), 0, 255).astype(np.uint8)
cv2.imwrite('sample_tcc_images/tcc_cyclone.png', img1)

# 2. Normal Rain (smaller, moderately cold core ~ 80)
img2 = generate_blob(size=64, center=(25,40), sigma=7, min_val=80)
img2 = np.clip(img2 + np.random.normal(0, 5, img2.shape), 0, 255).astype(np.uint8)
cv2.imwrite('sample_tcc_images/tcc_rain.png', img2)

# 3. No TCC (warm background, no cold spots)
img3 = np.clip(np.random.normal(190, 10, (64, 64)), 0, 255).astype(np.uint8)
cv2.imwrite('sample_tcc_images/notcc_clear.png', img3)

print("Images generated successfully in 'sample_tcc_images' folder.")
