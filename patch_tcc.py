import os

def main():
    with open("TCC.py", "r", encoding="utf-8") as f:
        code = f.read()
        
    func_code = """
def extract_tcc_parameters(img):
    from scipy.ndimage import label, center_of_mass
    import numpy as np
    binary = np.where(img < 100, 1, 0).astype(np.uint8)
    labeled_array, num_features = label(binary)
    clusters = []
    for cluster_id in range(1, num_features + 1):
        cluster_mask = labeled_array == cluster_id
        coords = np.argwhere(cluster_mask)
        if coords.size == 0: continue
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        Tb_values = img[y_coords, x_coords].astype(np.float32)
        min_Tb = float(Tb_values.min())
        mean_Tb = float(Tb_values.mean())
        median_Tb = float(np.median(Tb_values))
        center_y, center_x = center_of_mass(cluster_mask)
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        min_radius = float(distances.min())
        max_radius = float(distances.max())
        mean_radius = float(distances.mean())
        cloud_top_height = (290.0 - min_Tb) * 0.2
        clusters.append({
            'PixelCount': len(coords),
            'MeanTb': round(mean_Tb, 2),
            'MinTb': round(min_Tb, 2),
            'MedianTb': round(median_Tb, 2),
            'CenterX': round(center_x, 2),
            'CenterY': round(center_y, 2),
            'MinRadius': round(min_radius, 2),
            'MaxRadius': round(max_radius, 2),
            'MeanRadius': round(mean_radius, 2),
            'CloudTopHeight': round(cloud_top_height, 2)
        })
    return clusters

def predict_and_extract(image_array):
"""
    
    # Replace predict_and_extract definition to insert the new function above it
    if "def extract_tcc_parameters" not in code:
        code = code.replace("def predict_and_extract(image_array):", func_code)
        with open("TCC.py", "w", encoding="utf-8") as f:
            f.write(code)
            
if __name__ == "__main__":
    main()
