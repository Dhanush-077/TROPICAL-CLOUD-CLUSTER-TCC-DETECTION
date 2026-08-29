import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import label, center_of_mass
import gradio as gr

# Unbuffer stdout
sys.stdout.reconfigure(line_buffering=True)

# Suppress TensorFlow verbose logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras

# Determine paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "tcc_classifier_model.keras")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT_DIR, "TROPICAL-CLOUD-CLUSTER-TCC-DETECTION-main", "tcc_classifier_model.keras")

print(f"Loading TCC Detector model from: {MODEL_PATH}", flush=True)
model = keras.models.load_model(MODEL_PATH)
print("[OK] TCC Classifier Model loaded successfully!", flush=True)

def extract_tcc_parameters(img_gray):
    # Temperature threshold: pixels < 100 correspond to cold convective cloud tops
    binary = np.where(img_gray < 100, 1, 0).astype(np.uint8)
    labeled_array, num_features = label(binary)
    clusters = []
    
    for cluster_id in range(1, num_features + 1):
        cluster_mask = (labeled_array == cluster_id)
        coords = np.argwhere(cluster_mask)
        if coords.size < 5:  # Ignore tiny noise artifacts
            continue
        
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        Tb_values = img_gray[y_coords, x_coords].astype(np.float32)
        min_Tb = float(Tb_values.min())
        mean_Tb = float(Tb_values.mean())
        median_Tb = float(np.median(Tb_values))
        center_y, center_x = center_of_mass(cluster_mask)
        
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        min_radius = float(distances.min())
        max_radius = float(distances.max())
        mean_radius = float(distances.mean())
        cloud_top_height = round((290.0 - min_Tb) * 0.2, 2)
        
        pixel_count = len(coords)
        severity = get_severity_label(pixel_count, min_Tb)
        
        clusters.append({
            "Cluster ID": f"Cluster #{len(clusters) + 1}",
            "Severity": severity,
            "Pixel Count": pixel_count,
            "Convective Center (X, Y)": f"({round(center_x, 1)}, {round(center_y, 1)})",
            "Min Tb (°C)": round(min_Tb, 1),
            "Mean Tb (°C)": round(mean_Tb, 1),
            "Median Tb (°C)": round(median_Tb, 1),
            "Min Radius (px)": round(min_radius, 1),
            "Max Radius (px)": round(max_radius, 1),
            "Mean Radius (px)": round(mean_radius, 1),
            "Cloud Top Height (km)": cloud_top_height
        })
    return clusters

def get_severity_label(size, min_tb):
    if size < 50 or min_tb > 120:
        return "Normal Cloud"
    elif size < 200:
        return "Rain Cluster"
    elif size < 500:
        return "Heavy Rain Cluster"
    elif size < 800:
        return "Storm System"
    else:
        return "Tropical Cyclone"

def annotate_image(img_gray):
    # Rescale to 256x256 display resolution for visual clarity
    h, w = img_gray.shape
    scale = 4 if max(h, w) <= 64 else 1
    display_gray = cv2.resize(img_gray, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    display_rgb = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2RGB)
    
    binary = np.where(display_gray < 100, 1, 0).astype(np.uint8)
    labeled_array, num_features = label(binary)
    
    for cluster_id in range(1, num_features + 1):
        mask = (labeled_array == cluster_id).astype(np.uint8)
        coords = np.argwhere(mask)
        if coords.size < (5 * scale * scale):
            continue
        
        # Green contour outline around detected cluster
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display_rgb, contours, -1, (0, 255, 0), 2)
        
        # Red center dot & crosshair at convective center (coldest core)
        cy, cx = center_of_mass(mask)
        cv2.circle(display_rgb, (int(cx), int(cy)), 4, (255, 0, 0), -1)
        cv2.drawMarker(display_rgb, (int(cx), int(cy)), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
        
    return display_rgb

def process_ir_image(image):
    if image is None:
        return "### No Image Provided\nPlease upload an IR satellite image to run detection.", None, pd.DataFrame()
    
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
        
    img_model = cv2.resize(img_gray, (64, 64))
    img_norm = img_model.reshape(1, 64, 64, 1) / 255.0
    
    prob = float(model.predict(img_norm, verbose=0)[0][0])
    is_tcc = prob > 0.5
    confidence_pct = round(prob * 100 if is_tcc else (1 - prob) * 100, 1)
    label_str = "TROPICAL CLOUD CLUSTER (TCC) DETECTED" if is_tcc else "NO TCC DETECTED"
    
    clusters = extract_tcc_parameters(img_model)
    annotated_img = annotate_image(img_model)
    
    summary_md = f"## Detection Results\n\n"
    if is_tcc:
        summary_md += f"### Status: `{label_str}`\n"
    else:
        summary_md += f"### Status: `{label_str}`\n"
        
    summary_md += f"- **Confidence Level**: `{confidence_pct}%` (Raw score: `{prob:.4f}`)\n"
    summary_md += f"- **Clusters Extracted**: `{len(clusters)}`\n"
    
    if is_tcc and clusters:
        df = pd.DataFrame(clusters)
        top_severity = clusters[0]["Severity"]
        summary_md += f"- **Maximum Cluster Severity**: `{top_severity}`\n"
    else:
        df = pd.DataFrame()
        summary_md += f"\n> *Note: Image analyzed. No convective cloud clusters meeting ISRO TCC brightness temperature threshold (< 100 Tb).* "
        
    return summary_md, annotated_img, df

# Sample images list if available
sample_dir = os.path.join(ROOT_DIR, "sample_tcc_images")
if not os.path.exists(sample_dir):
    sample_dir = os.path.join(ROOT_DIR, "TROPICAL-CLOUD-CLUSTER-TCC-DETECTION-main", "sample_tcc_images")

example_images = []
if os.path.exists(sample_dir):
    example_images = [
        [os.path.join(sample_dir, "tcc_cyclone.png")],
        [os.path.join(sample_dir, "tcc_rain.png")],
        [os.path.join(sample_dir, "notcc_clear.png")]
    ]

with gr.Blocks(title="ISRO TCC Detection Web Interface") as demo:
    gr.Markdown(
        """
        # Tropical Cloud Cluster (TCC) Detection System
        ### ISRO Hackathon Challenge #9 - INSAT Satellite IRBT Data Analysis
        Automated Deep Learning & Computer Vision pipeline to detect Tropical Cloud Clusters (TCCs), calculate ISRO-specified convective parameters (Convective Center, Radii, Brightness Temperature statistics, Cloud-Top Height), and classify severity levels.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="Upload Infrared (IR) Satellite Image")
            btn_analyze = gr.Button("Detect & Extract Parameters", variant="primary")
            
            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=input_img,
                    label="Sample Demo IR Images"
                )
                
        with gr.Column(scale=1):
            summary_output = gr.Markdown(label="Detection Summary")
            annotated_output = gr.Image(label="Annotated IR Cluster Boundaries & Convective Center")
            
    gr.Markdown("### Extracted ISRO TCC Cluster Parameters & Feature Table")
    df_output = gr.Dataframe(label="Cluster Feature Matrix")
    
    btn_analyze.click(
        fn=process_ir_image,
        inputs=[input_img],
        outputs=[summary_output, annotated_output, df_output]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[INFO] Launching Gradio Web Server on port {port}...", flush=True)
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        quiet=False
    )
