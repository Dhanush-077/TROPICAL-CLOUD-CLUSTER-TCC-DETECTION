# Deploying to Render

This project is configured to deploy on [Render](https://render.com) using the included `render.yaml` configuration file.

## Prerequisites

1. A GitHub account with this repository connected
2. A [Render account](https://dashboard.render.com/register) (free tier available)

## Deployment Steps

### 1. Connect GitHub to Render
- Go to [Render Dashboard](https://dashboard.render.com)
- Click "New +" → "Web Service"
- Select "Deploy existing repository" and connect your GitHub
- Select the `TROPICAL-CLOUD-CLUSTER-TCC-DETECTION` repository

### 2. Configure the Web Service
Render will automatically detect `render.yaml` and use its configuration. You just need to:
- Verify the name: `tcc-detection`
- Check the runtime: `Python 3.11`
- Ensure region is set to your preference

### 3. Deploy
- Click "Create Web Service"
- Render will automatically:
  - Pull dependencies from `requirements.txt`
  - Install TensorFlow, Keras, OpenCV, and other packages
  - Start the Gradio app on port 10000
  - Provide you with a public URL

### 4. Access Your App
Once deployment completes (5-10 minutes), your TCC Detection app will be available at:
```
https://tcc-detection.onrender.com
```

## Configuration Details

The `render.yaml` file specifies:
- **Runtime**: Python 3.11
- **Build Command**: Installs dependencies from requirements.txt
- **Start Command**: Launches `python app.py`
- **Port**: 10000 (Render's standard web service port)
- **Environment Variables**: TensorFlow optimization settings for inference

## Important Notes

- The app requires ~2.5GB disk space for TensorFlow and model files
- Initial deployment may take 5-10 minutes due to dependency installation
- Free tier has 0.5GB RAM; for production use, upgrade to a paid plan
- The model files are included in the repository

## Troubleshooting

### If deployment fails:
1. Check the "Logs" tab in Render dashboard
2. Ensure `requirements.txt` is in the root directory
3. Verify `render.yaml` is properly formatted
4. Check that the GitHub repository is up to date (`git push`)

### If app crashes after deployment:
- Check Render logs for TensorFlow errors
- Verify all model files are present in the repository
- Try redeploying with "Manual Deploy" button

## Custom Domain

To use a custom domain:
1. Go to Web Service settings
2. Click "Add Custom Domain"
3. Point your domain's CNAME to Render's provided URL
