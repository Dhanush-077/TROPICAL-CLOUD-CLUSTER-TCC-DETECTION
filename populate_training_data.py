import os
import numpy as np
import cv2

def generate_blob(size=64, center=(32,32), sigma=10, min_val=50):
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    img = 200 - (200 - min_val) * np.exp(- (dist**2) / (2.0 * sigma**2))
    return img.astype(np.uint8)

def main():
    tcc_dir = "../dataset/TCC"
    notcc_dir = "../dataset/NoTCC"
    
    os.makedirs(tcc_dir, exist_ok=True)
    os.makedirs(notcc_dir, exist_ok=True)
    
    print("Generating TCC training images...")
    for i in range(100):
        # random center and size
        cx, cy = np.random.randint(15, 49, 2)
        sigma = np.random.uniform(5, 15)
        min_val = np.random.uniform(20, 90)
        img = generate_blob(center=(cx, cy), sigma=sigma, min_val=min_val)
        img = np.clip(img + np.random.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(tcc_dir, f"train_tcc_{i}.png"), img)
        
    print("Generating NoTCC training images...")
    for i in range(100):
        # warm background, no blobs
        img = np.random.normal(190, 15, (64, 64))
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(notcc_dir, f"train_notcc_{i}.png"), img)

    print("Training data populated!")

if __name__ == "__main__":
    main()
