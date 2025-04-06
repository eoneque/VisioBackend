from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from deepface import DeepFace
import numpy as np
import supabase
from uuid import uuid4
import os
import json 
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import io
import pytesseract
from inference_sdk import InferenceHTTPClient
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import logging
from pydantic import BaseModel
import face_recognition
import whisper


import base64
import tempfile

os.makedirs("uploads", exist_ok=True)

SUPABASE_URL = "https://jiovwqbaqkgzztfwcqky.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imppb3Z3cWJhcWtnenp0ZndjcWt5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI1MTUzMDAsImV4cCI6MjA1ODA5MTMwMH0.ldnnoZc_11AZ3Iyk6URO1vDz2NZ-RQEhHTG_tZiZgas"


supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.post("/register/")
async def register(file: UploadFile = File(...), user_id: str = Form(...)):  
    img_path = f"uploads/{file.filename}"
    with open(img_path, "wb") as f:
        f.write(file.file.read())

    try:
        embedding = DeepFace.represent(img_path, model_name="Facenet512")[0]["embedding"]
        print(f"Embedding dimensions: {len(embedding)}")  # Debugging: Check dimensions
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate face embedding")

    face_id = str(uuid4())


    data = {
        "id": face_id,
        "user_id": user_id, 
        "face_vector": embedding
    }
    response = supabase_client.table("faces").insert(data).execute()


    if hasattr(response, "error") and response.error:
        raise HTTPException(status_code=500, detail="Failed to store face data")

    return {"uuid": face_id}

    
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    match: bool
    uuid: Optional[str]
    confidence: float

async def save_uploaded_file(file: UploadFile) -> str:
    """Save the uploaded file to the uploads directory and return the file path."""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    img_path = f"{upload_dir}/{file.filename}"
    try:
        with open(img_path, "wb") as f:
            f.write(await file.read())
        return img_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

def preprocess_image(img_path: str) -> str:
    """Preprocess the image to improve face detection."""
    try:
        img = Image.open(img_path)
        # Convert to grayscale
        img = img.convert("L")
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        # Resize image
        img = img.resize((160, 160))
        # Save the preprocessed image
        preprocessed_path = img_path.replace(".", "_preprocessed.")
        img.save(preprocessed_path)
        return preprocessed_path
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=500, detail="Failed to preprocess image")

async def generate_face_embedding(img_path: str, retries: int = 3) -> Optional[np.ndarray]:
    """Generate face embedding using DeepFace with retries."""
    for attempt in range(retries):
        try:
            embedding = np.array(DeepFace.represent(img_path, model_name="Facenet512")[0]["embedding"], dtype=np.float32)
            logger.info(f"Generated embedding dimensions: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                # Preprocess the image and try again
                img_path = preprocess_image(img_path)
                continue
            else:
                logger.error(f"Error generating embedding after {retries} attempts: {e}")
                raise HTTPException(status_code=500, detail="Failed to generate face embedding")

async def fetch_stored_faces() -> List[Dict]:
    """Fetch stored face embeddings from the database."""
    try:
        response = supabase_client.table("faces").select("*").execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching faces from database: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch face data")

def calculate_similarity(stored_embedding: np.ndarray, new_embedding: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    similarity = np.dot(stored_embedding, new_embedding) / (np.linalg.norm(stored_embedding) * np.linalg.norm(new_embedding))
    return float(similarity)

@app.post("/verify/", response_model=VerificationResult)
async def verify(file: UploadFile):
    """Verify if the uploaded face matches any stored face."""
    # Save the uploaded file
    img_path = await save_uploaded_file(file)

    # Generate face embedding with retries
    new_embedding = await generate_face_embedding(img_path)

    # Fetch stored faces
    faces = await fetch_stored_faces()

    if not faces:
        return VerificationResult(match=False, confidence=0.0)

    # Find the best match
    best_match = None
    best_confidence = 0.0
    for face in faces:
        try:
            stored_embedding = np.array(json.loads(face["face_vector"]), dtype=np.float32)
            similarity = calculate_similarity(stored_embedding, new_embedding)
            logger.info(f"Similarity with {face['id']}: {similarity}")

            if similarity > best_confidence:
                best_confidence = similarity
                best_match = face["user_id"]
        except Exception as e:
            logger.error(f"Error processing face {face['id']}: {e}")
            continue

    # Determine if the match is above the threshold
    match = best_confidence >= 0.5  # Adjusted threshold
    return VerificationResult(match=match, uuid=best_match, confidence=best_confidence)




model = YOLO("yolov8n.pt")  


# classroom_objects = {"person", "chair", "table", "laptop", "whiteboard", "blackboard", "book", "bag"}

# @app.post("/detect")
# async def detect_object(file: UploadFile = File(...)):
#     # Read the image
#     image = Image.open(io.BytesIO(await file.read())).convert("RGB")
#     image = np.array(image)

#     results = model(image)[0]

#     detected_objects = set()

#     for box in results.boxes.data:
#         x1, y1, x2, y2, confidence, class_id = box.tolist()
#         obj_name = model.names[int(class_id)]

#         # Filter only classroom objects
#         if obj_name in classroom_objects:
#             detected_objects.add(obj_name)

#     return {"objects_detected": list(detected_objects) if detected_objects else ["No classroom objects detected"]}



CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="G3vJGSbfkKADKXqnq3IM"
)

@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    try:
        # Read and resize image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Resize image to a smaller size (e.g., 800px max dimension)
        max_size = 800
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Save resized image
        image_path = f"temp_{file.filename}"
        resized_image.save(image_path, quality=85, optimize=True)

        # Perform inference
        result = CLIENT.infer(image_path, model_id="visio-kcc4u/2")

        # Clean up temporary file
        os.remove(image_path)

        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
    



@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
 
        contents = await image.read()
        image_data = io.BytesIO(contents)
        img = Image.open(image_data)

        extracted_text = pytesseract.image_to_string(img)

        return JSONResponse(content={"text": extracted_text.strip()})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")




@app.post("/registers/")
async def register(file: UploadFile = File(...), user_id: str = Form(...)):
    img_path = f"uploads/{file.filename}"
    with open(img_path, "wb") as f:
        f.write(await file.read())

    try:
        # Load and encode face
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # Get face encoding
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        
        # Pad the encoding to 512 dimensions
        padded_encoding = np.zeros(512)
        padded_encoding[:128] = face_encoding  # Copy the 128 dimensions and pad with zeros
        
        face_id = str(uuid4())

        data = {
            "id": face_id,
            "user_id": user_id,
            "face_vector": padded_encoding.tolist()
        }
        response = supabase_client.table("faces").insert(data).execute()

        if hasattr(response, "error") and response.error:
            raise HTTPException(status_code=500, detail="Failed to store face data")

        return {"uuid": face_id}
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Failed to register face")

@app.post("/verifys/", response_model=VerificationResult)
async def verify(file: UploadFile):
    img_path = f"uploads/{file.filename}"
    with open(img_path, "wb") as f:
        f.write(await file.read())

    try:
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return VerificationResult(match=False, uuid=None, confidence=0.0)
            
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]

        faces = await fetch_stored_faces()
        if not faces:
            return VerificationResult(match=False, uuid=None, confidence=0.0)

        best_match = None
        best_confidence = 0.0

        for face in faces:
            try:
                # Parse the stored vector properly
                if isinstance(face["face_vector"], str):
                    stored_encoding = np.array(json.loads(face["face_vector"]), dtype=np.float64)
                else:
                    stored_encoding = np.array(face["face_vector"], dtype=np.float64)
                
                # Ensure the stored encoding is 1-dimensional and has correct length
                if stored_encoding.size < 128:
                    logger.warning(f"Invalid vector size for face {face['id']}")
                    continue
                
                # Compare only the first 128 dimensions
                stored_encoding = stored_encoding[:128].reshape(128)
                face_distances = face_recognition.face_distance([stored_encoding], face_encoding)
                confidence = 1 - face_distances[0]

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = face["user_id"]
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for face {face['id']}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error comparing faces: {e}")
                continue

        match = best_confidence >= 0.6
        return VerificationResult(match=match, uuid=best_match, confidence=float(best_confidence))

    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify face")




model = whisper.load_model("base")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Transcribe audio
    result = model.transcribe(file_path)

    # Remove temp file
    os.remove(file_path)

    return {"text": result["text"]}