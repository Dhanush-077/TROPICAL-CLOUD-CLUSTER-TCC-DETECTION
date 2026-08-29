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

### Always start here: Check the logs
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Select your `tcc-detection` service
3. Click the failed "Deploy" in the Events feed to view build logs
4. Or click "Logs" tab to view runtime errors
5. Search for `error` to find the root cause

### Common Issues & Solutions

#### Build Fails with "ModuleNotFoundError"
**Problem:** `ModuleNotFoundError: No module named 'tensorflow'` or similar
- **Solution:** 
  - Verify `requirements.txt` is in the repository root
  - Check that all dependencies are spelled correctly (case-sensitive)
  - Ensure `requirements.txt` has no comments or syntax errors
  - On Windows, file names must match exactly (tensorflow ≠ TensorFlow in imports)

#### Build Fails with "requires Python >= X.Y"
**Problem:** TensorFlow dependency error about Python version
- **Solution:**
  - Check `render.yaml` specifies `runtime: python311` (Python 3.11)
  - If you need a different Python version, update `render.yaml`:
    ```yaml
    runtime: python310  # or python39, python312
    ```

#### App Starts but Returns 502 Bad Gateway
**Problem:** App crashes immediately after starting
- **Solution:**
  - Check Render logs for TensorFlow or Keras errors
  - Verify model files exist: `tcc_classifier_model.keras`
  - Check port binding in `app.py` uses `PORT` environment variable (already configured)
  - TensorFlow can take 2-3 minutes to load; wait before visiting URL

#### App Takes Too Long to Load (timeout)
**Problem:** First request times out or hangs
- **Solution:**
  - This is normal with TensorFlow (large library to initialize)
  - Wait 3-5 minutes after deployment before testing
  - Model loading happens on first request
  - Subsequent requests will be fast
  - Check logs for `[OK] TCC Classifier Model loaded successfully!`

#### Model File Not Found
**Problem:** Error: "tcc_classifier_model.keras: No such file or directory"
- **Solution:**
  - Verify model file is committed to GitHub: `git add tcc_classifier_model.keras`
  - Check file size isn't too large (>100MB may fail on free tier)
  - If file is missing, get it from your local directory and commit

#### Out of Memory or Disk Space
**Problem:** "No space left on device" or memory errors
- **Solution:**
  - Free tier has 0.5GB RAM and ~3GB storage
  - Upgrade to a paid plan ($7+/month)
  - Or clear Render cache: Redeploy with "Clear build cache" option

#### File Path Errors (Windows vs Linux)
**Problem:** `FileNotFoundError` for paths with backslashes
- **Solution:**
  - Use forward slashes in Python: `path/to/file` not `path\to\file`
  - Already handled in `app.py` with `os.path.join()`
  - No changes needed

### Quick Redeploy
1. Go to your service in Render Dashboard
2. Click "Manual Deploy" button
3. Select "Clear build cache" option
4. Click "Deploy latest commit"

### Enable Debug Logging
Add this to your environment variables in Render Dashboard:
```
TF_CPP_MIN_LOG_LEVEL = 0
```
Then recheck logs for more detailed TensorFlow output.

### Still Having Issues?
- Check [Render's official troubleshooting guide](https://render.com/docs/troubleshooting-deploys)
- View full logs in Render Dashboard's Log Explorer
- Ensure your GitHub repository is synced: `git push origin main`

## Custom Domain

To use a custom domain:
1. Go to Web Service settings
2. Click "Add Custom Domain"
3. Point your domain's CNAME to Render's provided URL
4. Wait 5-60 minutes for DNS propagation
