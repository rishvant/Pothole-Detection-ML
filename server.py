import os
import json
import logging
from pathlib import Path
from typing import NamedTuple

from fastapi import FastAPI, File, Form, UploadFile, WebSocket
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Table, Column, Float, MetaData, String
from sqlalchemy.orm import sessionmaker
import numpy as np
import cv2
from ultralytics import YOLO
from sqlalchemy import select

# Initialize app and logger
app = FastAPI()
logger = logging.getLogger("RoadDamageServer")
logger.setLevel(logging.INFO)

# Database setup
database_url = "sqlite:///./detections.db"
engine = create_engine(database_url, connect_args={"check_same_thread": False})
metadata = MetaData()
detection_table = Table(
    "detections", metadata,
    Column("threshold", Float),
    Column("damage_type", String),
    Column("latitude", Float),
    Column("longitude", Float)
)
metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Model setup
MODEL_LOCAL_PATH = "./models/YOLOv8_Small_RDD.pt"
net = YOLO(MODEL_LOCAL_PATH)

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: list  # Changed from np.ndarray to list
    latitude: float = None
    longitude: float = None



# Add a new endpoint to fetch the detections from the database
@app.get("/detections/")
async def get_detections():
    db = SessionLocal()
    try:
        # Query all records from the detection table
        query = select(detection_table)
        results = db.execute(query).fetchall()
        
        # Convert the results to a list of dictionaries
        detections = []
        for row in results:
            detection = {
                "threshold": row.threshold,
                "damage_type": row.damage_type,
                "latitude": row.latitude,
                "longitude": row.longitude
            }
            detections.append(detection)
        
        return {"detections": detections}
    except Exception as e:
        return {"error": f"Failed to retrieve detections: {str(e)}"}
    finally:
        db.close()
@app.post("/upload_frame/")
async def upload_frame(file: UploadFile, threshold: float = Form(...), latitude: float = Form(None), longitude: float = Form(None)):
    try:
        # Read the uploaded image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image with YOLO
        image_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
        results = net.predict(image_resized, conf=threshold)

        detections = []
        db_detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for _box in boxes:
                detection = Detection(
    class_id=int(_box.cls),
    label=CLASSES[int(_box.cls)],
    score=float(_box.conf),
    box=_box.xyxy[0].astype(int).tolist(),  # Convert ndarray to list
    latitude=latitude,
    longitude=longitude
)

                detections.append(detection)

                # Save detection in DB
                db_detections.append({
                    "threshold": threshold,
                    "damage_type": detection.label,
                    "latitude": latitude,
                    "longitude": longitude
                })

        # Store detections in the database
        db = SessionLocal()
        try:
            for record in db_detections:
                db.execute(detection_table.insert().values(**record))
            db.commit()
        finally:
            db.close()

        return JSONResponse(content={"detections": [det._asdict() for det in detections]})

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return JSONResponse(content={"error": "Failed to process frame"}, status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            threshold = message.get("threshold", 0.5)
            frame_data = np.frombuffer(bytearray(message.get("frame", "")), dtype=np.uint8)
            image = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            image_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
            results = net.predict(image_resized, conf=threshold)
            response_data = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for _box in boxes:
                    detection = {
                        "class_id": int(_box.cls),
                        "label": CLASSES[int(_box.cls)],
                        "score": float(_box.conf),
                        "box": _box.xyxy[0].tolist()
                    }
                    response_data.append(detection)

            await websocket.send_json(response_data)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed")