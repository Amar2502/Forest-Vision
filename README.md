# ForestVision 🌳

**ForestVision** is an AI-powered web application that tracks deforestation and forest area changes using real-time satellite imagery from Sentinel-2. The application uses deep learning to segment forest areas and provides detailed metrics and visualizations of forest cover changes over time.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

## Features

- 🌍 **Global Coverage**: Analyze forest changes anywhere on Earth using Sentinel-2 satellite data
- 🤖 **AI-Powered Segmentation**: Uses Attention U-Net deep learning model to accurately identify forest areas
- 📊 **Detailed Metrics**: Calculate forest cover percentage, area in hectares, and deforestation rates
- 📅 **Time-Series Analysis**: Track forest changes between any two dates from 2017 to 2024
- 🗺️ **Interactive Maps**: Visualize locations using interactive maps with coordinate input
- 📈 **Visual Analytics**: View satellite images, forest segmentation overlays, and deforestation trends
- 🔍 **Multi-Sample Analysis**: Analyze incremental changes with multiple time intervals between start and end dates
- ☁️ **Cloud-Filtered**: Automatically selects cloud-free or low-cloud-coverage satellite images

## How It Works

### Process Flow

1. **User Input**: User enters coordinates (latitude/longitude) and selects date range via Streamlit UI
2. **Bounding Box**: Backend calculates 512×512 pixel area (5.12 km × 5.12 km = 2,621.44 hectares) centered on coordinates
3. **Image Retrieval**: Queries SentinelHub Catalog API to find Sentinel-2 L2A images within ±180 days of requested dates
4. **Cloud Filtering**: Selects images with <10% cloud coverage (falls back to <50% if needed) using penalty algorithm: `penalty = 10 × cloud_coverage + |days_from_requested_date|`
5. **Image Preprocessing**: Downloads RGB (B02, B03, B04) and 4-band (RGB + NIR B08) images, normalizes to 0-1 range
6. **AI Segmentation**: Pre-trained Attention U-Net model processes 4-band images, outputs probability map (0-1), applies 0.7 threshold to create binary mask (white=forest, black=non-forest)
7. **Metrics Calculation**: 
   - Forest cover % = `(forest_pixels / total_pixels) × 100`
   - Area (hectares) = `(coverage % / 100) × 2,621.44`
   - Change detection = `coverage_end - coverage_start`
   - Monthly loss rate = `area_change / months_between_images`
8. **Visualization**: Creates colored overlays (green=forest, red=deforestation) composited on satellite images

### Architecture

**Backend (FastAPI)**:
- `sentinelhub_requester.py`: Handles SentinelHub API communication, image search, and download requests
- `image_array_loader.py`: Orchestrates image loading for multiple dates, handles cloud filtering
- `segmenter.py`: Runs AI model inference for forest segmentation
- `main.py`: FastAPI endpoints (`/`, `/test_coordinates`, `/get_satellite_images`)

**Frontend (Streamlit)**:
- `ForestVision.py`: Main UI, state management, user interaction
- `utils/metrics_processing.py`: Calculates forest cover statistics, converts to hectares
- `utils/graphics_processing.py`: Creates overlays, colorizes masks, smooths images
- `utils/response_parsing.py`: Parses API responses, decodes base64 images
- `utils/map_injection.py`: Displays interactive PyDeck map

## Installation

### Prerequisites
- Python 3.8+
- SentinelHub account (free tier available)
- 4GB+ RAM

### Steps

1. **Clone repository**:
   ```bash
   git clone <https://github.com/Amar2502/Forest-Vision.git
   cd forestvision-master
   ```

2. **Create virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Note: `requirements.txt` includes platform-specific TensorFlow (macOS vs Windows/Linux)

4. **Verify model file**: Ensure `backend/model_ressources/att_unet_4b.hdf5` exists

## Configuration

### SentinelHub Credentials

1. Create account at [sentinel-hub.com](https://www.sentinel-hub.com/)
2. Navigate to **User Settings** → **OAuth clients**
3. Create OAuth client and copy **Client ID** and **Client Secret**
4. Create `.env` file in project root:
   ```env
   SENTINEL_CLIENT_ID=your_client_id_here
   SENTINEL_CLIENT_SECRET=your_client_secret_here
   ```
5. **Important**: `.env` is in `.gitignore` - never commit credentials

## Usage

### Starting the Application

1. **Start Backend** (Terminal 1):
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
   ```
   Verify: Visit `http://localhost:8080` → should show `{"api status": "running"}`

2. **Start Frontend** (Terminal 2):
   ```bash
   streamlit run frontend/ForestVision.py
   ```
   App opens at `http://localhost:8501`

### Using the Application

**Method 1: Manual Input**
1. Enter coordinates (latitude/longitude) in sidebar
2. Click "View on map" to verify location
3. Select start and end dates (2017-2024)
4. Click "Calculate forest loss"
5. Wait for processing (30 seconds - 2 minutes)

**Method 2: Example Locations**
- Click "Calculate" on Bolivia or Brazil example cards (pre-configured coordinates and dates)

**Results Display**:
- **Summary**: Actual dates found and percentage change
- **Start/End Date Columns**: Satellite images with forest overlays (green)
- **Total Forest Change**: Deforestation overlay (red) on end-date image
- **Metrics**: Forest cover percentages and hectares, with tabs for percent and hectare views

**Detailed Analysis** (Optional):
- Set sample number slider (2-8 intervals)
- Click "Calculate" to see time-series progression
- View charts: "Coverage Lost Since Start" and "Monthly Loss Rate"
- Review detailed metrics table

## Screenshots

### Screenshot 1: Main Application Interface

![Main Interface](screenshots/Screenshot%202025-12-05%20182316.png)

Shows the main application interface with:
- **Sidebar**: Coordinate inputs, date selectors, example location buttons (Bolivia/Brazil), detailed analysis controls
- **Main Area**: Interactive PyDeck map centered on selected coordinates

### Screenshot 2: Area Selection

![Analysis Results](screenshots/Screenshot%202025-12-05%20182424.png)

Displays analysis results with:
- **Summary**: Date range and percentage change
- **Three Columns**:
  - Left: Start/End date satellite images with forest segmentation overlays (green)
  - Middle: Total forest change visualization with deforestation overlay (red) and legend
  - Right: Metrics panel showing forest cover percentages and hectares (with tabs)

### Screenshot 3: Detailed Analysis

![Time-Series Analysis](screenshots/Screenshot%202025-12-05%20182507.png)

Shows detailed time-series analysis featuring:
- **Time-Series Images**: Multiple columns showing forest progression over time (satellite images + segmentation)
- **Charts**: "Coverage Lost Since Start" (cumulative loss %) and "Monthly Loss Rate" (hectares/month)
- **Metrics Table**: Detailed statistics including coverage %, time differences, relative changes, area changes, and monthly loss rates

## Project Structure

```
forestvision-master/
├── backend/
│   ├── main.py                 # FastAPI app and endpoints
│   ├── segmenter.py            # AI model inference
│   ├── sentinelhub_requester.py # SentinelHub API integration
│   ├── image_array_loader.py   # Image loading orchestration
│   ├── custom_layer.py         # Custom Keras layer
│   ├── utils.py                # Utility functions
│   └── model_ressources/
│       └── att_unet_4b.hdf5    # Pre-trained model
├── frontend/
│   ├── ForestVision.py         # Main Streamlit app
│   ├── images/                 # Static images (logo, examples)
│   └── utils/
│       ├── map_injection.py    # Map visualization
│       ├── html_injection.py   # HTML/CSS styling
│       ├── metrics_processing.py # Metrics calculations
│       ├── response_parsing.py  # API response parsing
│       └── graphics_processing.py # Image overlays
├── screenshots/                # Application screenshots
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Technologies

**Backend**:
- **FastAPI**: REST API framework
- **TensorFlow/Keras**: Deep learning for forest segmentation
- **SentinelHub SDK**: Access to Sentinel-2 satellite data
- **NumPy**: Image array operations
- **Pillow**: Image processing
- **Uvicorn**: ASGI server

**Frontend**:
- **Streamlit**: Web app framework
- **PyDeck**: Interactive map visualization
- **OpenCV**: Image filtering and processing
- **Pandas**: Data manipulation for metrics
- **Requests**: HTTP client for API calls

**AI/ML**:
- **Attention U-Net**: Pre-trained semantic segmentation model
- **Custom Keras Layer**: RepeatElements layer for model architecture

## API Documentation

### Base URL
```
http://localhost:8080
```

### Endpoints

#### `GET /`
Health check. Returns: `{"api status": "running"}`

#### `GET /test_coordinates`
Test image availability without full processing.

**Parameters**: `latitude`, `longitude`, `start_date`, `end_date`

**Response**:
```json
{
  "coordinates": {"latitude": "-8.49", "longitude": "-55.26"},
  "date_range": {"start": "2020-01-01", "end": "2023-12-31"},
  "credentials_loaded": true,
  "start_date_found": true,
  "end_date_found": true,
  "start_date_actual": "2020-05-13",
  "end_date_actual": "2023-12-15",
  "start_cloud_cover": 5.2,
  "end_cloud_cover": 8.7
}
```

#### `GET /get_satellite_images`
Main endpoint for forest analysis.

**Parameters**:
- `start_timeframe`, `end_timeframe`: Date range (YYYY-MM-DD)
- `latitude`, `longitude`: Coordinates
- `sample_number`: Number of time intervals (2-8, default: "2")
- `send_orginal_images`: Include original images ("True"/"False")

**Response**:
```json
{
  "date_list_loaded": ["2020-05-13", "2024-05-30"],
  "segmented_img_list": ["base64_encoded_image1", "base64_encoded_image2"],
  "original_img_list": ["base64_encoded_image1", "base64_encoded_image2"]
}
```

**Error Responses**: `400` (failed to load), `404` (no images found)

## Troubleshooting

### SentinelHub Credentials Not Found
- Verify `.env` file exists with `SENTINEL_CLIENT_ID` and `SENTINEL_CLIENT_SECRET`
- Restart backend after setting environment variables
- Check credentials in SentinelHub dashboard

### No Images Found
- Try wider date range (app searches ±180 days)
- Select different location
- Use dates between 2017-2024
- Test with `/test_coordinates` endpoint first

### Model File Not Found
- Verify `backend/model_ressources/att_unet_4b.hdf5` exists
- Check file permissions
- Re-download if corrupted

### Port Already in Use
- Find process: `netstat -ano | findstr :8080` (Windows) or `lsof -i :8080` (macOS/Linux)
- Change port: `uvicorn backend.main:app --port 8081`
- Update `API_URL` in `ForestVision.py` if changing port

### TensorFlow Import Errors
- Verify correct version: `tensorflow-macos` (macOS) or `tensorflow` (Windows/Linux)
- Reinstall: `pip install --upgrade -r requirements.txt`
- Check Python version (requires 3.8+)

### Streamlit App Not Loading
- Verify backend running: `http://localhost:8080`
- Check `API_URL` in `ForestVision.py` points to correct backend
- Check browser console for errors
- Verify all dependencies installed

### Slow Processing
- **Expected times**: 30-120 seconds for 2 images (CPU), faster with GPU
- Reduce `sample_number` for faster results
- Check internet connection speed
- Processing time increases with number of samples

---

**Made with 🌳 for forest conservation**
