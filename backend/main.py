import os
import io
import base64
import datetime as dt
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from backend.custom_layer import RepeatElements
from backend.segmenter import segment
from backend.image_array_loader import load_images_from_satellite

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

def numpy_to_base64(img_array):
    img_bytes = io.BytesIO()
    np.save(img_bytes, img_array)
    img_bytes.seek(0)
    img_b64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    return img_b64

app = FastAPI()

@app.get('/')
def index():
    return {'api status': "running"}

@app.get("/test_coordinates")
def test_coordinates(
    latitude: str = "-8.49",
    longitude: str = "-55.26",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31"
):
    """Test endpoint to check if images are available for given coordinates and date range."""
    from backend.sentinelhub_requester import create_sentinelhub_token, create_bounding_box, search_optimal_l2a_tiles
    from backend.utils import timeframe_constructor
    
    try:
        config, catalog = create_sentinelhub_token()
        
        # Check credentials
        has_credentials = bool(config.sh_client_id and config.sh_client_secret)
        
        # Create bounding box
        bbox = create_bounding_box(float(latitude), float(longitude))
        
        # Test search for start date
        start_results = search_optimal_l2a_tiles(
            catalog=catalog,
            bbox=bbox,
            date_request=start_date,
            range_days=180,  # Wider range
            max_cloud_coverage=50  # More lenient
        )
        
        # Test search for end date
        end_results = search_optimal_l2a_tiles(
            catalog=catalog,
            bbox=bbox,
            date_request=end_date,
            range_days=180,
            max_cloud_coverage=50
        )
        
        return {
            "coordinates": {"latitude": latitude, "longitude": longitude},
            "date_range": {"start": start_date, "end": end_date},
            "credentials_loaded": has_credentials,
            "start_date_found": start_results is not None,
            "end_date_found": end_results is not None,
            "start_date_actual": start_results.get('date') if start_results else None,
            "end_date_actual": end_results.get('date') if end_results else None,
            "start_cloud_cover": start_results.get('cloud_cover') if start_results else None,
            "end_cloud_cover": end_results.get('cloud_cover') if end_results else None,
            "bbox": str(bbox),
            "recommendations": []
        }
    except Exception as e:
        return {
            "error": str(e),
            "coordinates": {"latitude": latitude, "longitude": longitude},
            "date_range": {"start": start_date, "end": end_date}
        }

@app.get("/get_satellite_images")
def get_satellite_images(
    start_timeframe: str = "2020-05-13",
    end_timeframe: str = "2024-05-30",
    longitude: str = "-55.26209",
    latitude: str = "-8.48638",
    sample_number: str = "2",
    send_orginal_images = 'False'
    ):
    """Takes start and end date and coordinates and returns
    a JSON response object including dates, segmented images and optionally,
    original images. If sample_number equals two, only data belonging to the
    closest point in time to the start and the end date respectively will be
    returned, higher sample_numbers will return data from points in time between
    the start and the end date, as evenly spaced as possible.
    """
    start_dt = dt.datetime.strptime(start_timeframe, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(end_timeframe, "%Y-%m-%d")
    date_step_size = (end_dt - start_dt)/(int(sample_number) - 1)
    date_list = [
        dt.date.strftime(start_dt + i * date_step_size, "%Y-%m-%d")
        for i in range(int(sample_number))
    ]

    try:
        loaded_dates, img_arrays = load_images_from_satellite(
            lat_deg=float(latitude),
            lon_deg=float(longitude),
            date_list=date_list
        )
    except Exception as e:
        print(f"Error loading images from satellite: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to load satellite images: {str(e)}"}
        )

    if not loaded_dates or len(loaded_dates) == 0:
        error_msg = (
            f"No suitable cloud-free images found for coordinates "
            f"({latitude}, {longitude}) between {start_timeframe} and {end_timeframe}. "
            f"Please try a different location or wider date range."
        )
        print(error_msg)
        return JSONResponse(
            status_code=404,
            content={"error": error_msg}
        )

    number_of_dates = len(loaded_dates)
    original_img_arrays = []
    combined_img_arrays = []
    # Join 3- with 4-band, normalize 4-band channel to values between 0 and 1
    for date in range(number_of_dates):
        band_4_at_date = img_arrays[(date + number_of_dates)][:,:,3]
        vis_at_date = img_arrays[date]
        rescaled_vis_at_date = (vis_at_date * 255).astype(np.uint8)
        original_img_arrays.append(rescaled_vis_at_date)
        max_value = np.max(band_4_at_date)
        band_4_at_date = band_4_at_date / max_value
        combined_img_arrays.append(np.dstack((vis_at_date, band_4_at_date)))

    # Get model path - works whether running from project root or backend directory
    model_path = os.path.join(os.path.dirname(__file__), 'model_ressources', 'att_unet_4b.hdf5')
    if not os.path.exists(model_path):
        # Try relative to project root
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend', 'model_ressources', 'att_unet_4b.hdf5')
    
    model = load_model(
        filepath=model_path,
        custom_objects={'RepeatElements': RepeatElements}
    )
    segmented_img_arrays = [
        segment(img_at_date, model, threshold=0.7)
        for img_at_date in combined_img_arrays
    ]

    for i, arr in enumerate(segmented_img_arrays):
        if arr.shape != (512, 512):
            loaded_dates.pop(i)
            original_img_arrays.pop(i)
            segmented_img_arrays.pop(i)

    segmented_img_b64_list = [
        numpy_to_base64(img)
        for img in segmented_img_arrays
    ]
    original_img_b64_list = [
        numpy_to_base64(img)
        for img in original_img_arrays
    ]

    if send_orginal_images == 'False':
        json_response_content = {
                "date_list_loaded": loaded_dates,
                "segmented_img_list": segmented_img_b64_list}
    elif send_orginal_images == 'True':
        json_response_content = {
                "date_list_loaded": loaded_dates,
                "segmented_img_list": segmented_img_b64_list,
                "original_img_list": original_img_b64_list}
    return JSONResponse(content=json_response_content)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
