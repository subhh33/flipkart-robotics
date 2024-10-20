from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import cv2
import threading
from detect_objects import process_image

app = FastAPI()

# Mount the static directory to serve CSS, JS, and images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Shared variables
object_count = 0
object_details = ''
detection_thread = None
lock = threading.Lock()

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def run_object_detection():
    global object_count, object_details

    # Open the webcam
    cap = cv2.VideoCapture(1)  # Change to 0 if necessary
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to stop the object detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Process the frame and update global variables
        count, details = process_image(frame)

        # Use lock to safely update shared variables
        with lock:
            object_count = count
            object_details = details

        # Display the frame
        cv2.imshow("Real-Time Object Detection", frame)

        # Stop the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting real-time object detection.")
            break

    # Release the webcam after exiting the loop
    cap.release()
    cv2.destroyAllWindows()

@app.post("/object_detection_started")
async def start_process():
    global detection_thread
    
    # Start the detection loop in a separate thread
    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = threading.Thread(target=run_object_detection)
        detection_thread.start()
        return JSONResponse({"message": "Object detection started!"})
    
    return JSONResponse({"message": "Object detection is already running!"}, status_code=400)

@app.get("/object_detection", response_class=HTMLResponse)
async def get_information(request: Request):
    # Access shared variables safely
    with lock:
        count = object_count
        details = object_details
    
    return templates.TemplateResponse("information.html", {
        "request": request,
        "object_count": count,
        "object_details": details,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
