import os
import cv2
import numpy as np

def create_dummy_images():
    dirs = [
        "../dataset/TCC",
        "../dataset/NoTCC",
        "tcc_irbrt_images"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        for i in range(5):
            # 64x64 grayscale
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            # Make sure some are "cold" (<100)
            if "TCC" in d:
                img[20:40, 20:40] = 50
                
            path = os.path.join(d, f"dummy_image_{i}.png")
            cv2.imwrite(path, img)

if __name__ == "__main__":
    create_dummy_images()
