import base64
import cv2
import numpy as np
import io
import time
import uuid
import os
import math
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel 

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from sqlmodel import Session, select, text, func
from db import create_db_and_tables, engine, Violation, ViolationStatus, Camera

app = FastAPI(title="Safety Net IoT System")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
    allow_methods=["*"], allow_headers=["*"],
)

# --- FOLDER SCREENSHOT ---
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=SCREENSHOT_DIR), name="screenshots")

# --- GLOBAL CONFIG ---
SYSTEM_REGISTRATION_CODE = "123456" 

try:
    model = YOLO("best.pt")
    print("✅ Model best.pt loaded.")
except:
    print("⚠️ Using yolov8n.pt (fallback).")
    model = YOLO("yolov8n.pt")

# Config Deteksi & Threshold
DISPLAY_MIN_CONF = 0.30     # Tampilkan di layar jika yakin > 30%
HIGH_CONF_THRESHOLD = 0.85  # Anggap CRITICAL jika yakin > 85%

# Anti-Spam History
violator_history = {} 
SPATIAL_LOCK_RADIUS = 150 
MOVEMENT_THRESHOLD = 100 
ID_COOLDOWN = 30 

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# ==========================================
# 1. API REGISTRASI KAMERA (IOT STYLE)
# ==========================================
class CameraRegisterRequest(BaseModel):
    id: str
    name: str
    code: str

class CameraUpdateRequest(BaseModel):
    name: str
    code: str

@app.post("/api/cameras/register")
def register_camera(payload: CameraRegisterRequest):
    if payload.code != SYSTEM_REGISTRATION_CODE:
        raise HTTPException(status_code=401, detail="Kode Registrasi Salah!")
    with Session(engine) as session:
        existing_cam = session.get(Camera, payload.id)
        if existing_cam:
            return {"message": "Login berhasil.", "camera": existing_cam}
        new_cam = Camera(id=payload.id, location=payload.name, is_active=True)
        session.add(new_cam)
        session.commit()
        session.refresh(new_cam)
        return {"message": "Registrasi Berhasil!", "camera": new_cam}

@app.put("/api/cameras/{cam_id}")
def update_camera(cam_id: str, payload: CameraUpdateRequest):
    if payload.code != SYSTEM_REGISTRATION_CODE:
        raise HTTPException(status_code=401, detail="Password Salah!")
    with Session(engine) as session:
        camera = session.get(Camera, cam_id)
        if not camera: raise HTTPException(404, "Kamera tidak ditemukan")
        camera.location = payload.name
        session.add(camera)
        session.commit()
        return {"message": "Update berhasil", "camera": camera}

# ==========================================
# 2. WEBSOCKET (CORE DETECTION)
# ==========================================
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

@app.websocket("/yolo")
async def websocket_endpoint(websocket: WebSocket, cam_id: str):
    await websocket.accept()
    global violator_history 
    
    # Validasi Kamera
    with Session(engine) as session:
        camera = session.get(Camera, cam_id)
        if not camera:
            print(f"❌ Rejected: {cam_id} (Unregistered)")
            await websocket.close(code=4001)
            return
        print(f"✅ Connected: {cam_id}")

    try:
        while True:
            data = await websocket.receive_text()
            encoded = data.split(",", 1)[1] if "," in data else data
            
            try:
                np_arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None: continue

                # Inference
                results = model.track(frame, persist=True, verbose=False, conf=DISPLAY_MIN_CONF, iou=0.5)
                result = results[0]

                person_count = 0; helmet_count = 0
                current_frame_person_ids = []; boxes_data = []

                if result.boxes:
                    for box in result.boxes:
                        cls_name = model.names[int(box.cls[0])]
                        coords = box.xyxy[0].tolist()
                        conf_val = float(box.conf[0]) # Ambil Confidence Score
                        track_id = int(box.id[0]) if box.id is not None else None

                        # --- LOGIC CLASS ---
                        box_type = "person"
                        if 'helmet' in cls_name or 'hardhat' in cls_name: 
                            box_type = "helmet"; helmet_count += 1
                        elif 'head' in cls_name: 
                            box_type = "head" # Head = Merah (Bahaya)
                        elif 'person' in cls_name:
                            box_type = "person"; person_count += 1
                            if track_id is not None:
                                current_frame_person_ids.append({
                                    'id': track_id, 
                                    'center': ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2),
                                    'conf': conf_val 
                                })

                        # Kirim data visual ke Frontend (termasuk % Confidence)
                        label_str = f"{cls_name} {int(conf_val*100)}%"
                        boxes_data.append({"coords": coords, "label": label_str, "type": box_type})

                violation_count = max(0, person_count - helmet_count)
                screenshot_url = ""
                current_time = time.time()

                # --- SNAPSHOT & SAVE LOGIC ---
                if violation_count > 0:
                    for person in current_frame_person_ids:
                        p_id = person['id']
                        p_center = person['center']
                        p_conf = person['conf']
                        
                        should_take_shot = False
                        
                        # Tentukan Severity (CRITICAL vs WARNING)
                        severity_label = "CRITICAL" if p_conf >= HIGH_CONF_THRESHOLD else "WARNING"
                        
                        # Kunci History Unik per Kamera
                        history_key = f"{cam_id}_{p_id}" 

                        if history_key in violator_history:
                            last_data = violator_history[history_key]
                            # Logic Anti-Spam: Cooldown + Jarak
                            if (current_time - last_data['last_shot'] > ID_COOLDOWN) and (calculate_distance(p_center, last_data['last_center']) > MOVEMENT_THRESHOLD):
                                should_take_shot = True
                        else:
                            # Logic Anti-Ghosting
                            is_ghost = False
                            for k, h_data in violator_history.items():
                                if k.startswith(cam_id) and (calculate_distance(p_center, h_data['last_center']) < SPATIAL_LOCK_RADIUS) and (current_time - h_data['last_shot'] < ID_COOLDOWN):
                                    is_ghost = True; break
                            if not is_ghost: should_take_shot = True

                        # --- SIMPAN DATA JIKA VALID ---
                        if should_take_shot:
                            # 1. Simpan Gambar
                            filename = f"vio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cam_id}_ID{p_id}.jpg"
                            cv2.imwrite(os.path.join(SCREENSHOT_DIR, filename), frame)
                            screenshot_url = f"screenshots/{filename}"
                            
                            # 2. Simpan Database (Dengan CONFIDENCE)
                            with Session(engine) as session:
                                session.add(Violation(
                                    person_count=person_count, 
                                    helmet_count=helmet_count, 
                                    screenshot_path=screenshot_url, 
                                    confidence=p_conf,        # <--- CONFIDENCE DISIMPAN
                                    severity=severity_label,  # <--- SEVERITY DISIMPAN
                                    camera_id=cam_id
                                ))
                                session.commit()
                            
                            print(f"📸 SNAPSHOT [{cam_id}] ID:{p_id} {severity_label} ({int(p_conf*100)}%)")
                            
                            # Update History
                            violator_history[history_key] = {'last_center': p_center, 'last_shot': current_time}
                            break 

                if len(violator_history) > 100: violator_history.clear()
                
                await websocket.send_json({
                    "person": person_count, "helmet": helmet_count, 
                    "violation": violation_count, "boxes": boxes_data, "screenshot": screenshot_url
                })

            except Exception: continue
    except WebSocketDisconnect: print(f"{cam_id} Disconnected")

# ==========================================
# 3. API DASHBOARD & MANAGEMENT
# ==========================================
@app.get("/api/violations", response_model=dict)
def get_violations(page: int = 1, limit: int = 10, severity: Optional[str] = None, status: Optional[str] = None, camera_id: Optional[str] = None):
    with Session(engine) as session:
        query = select(Violation).order_by(Violation.timestamp.desc())
        if severity: query = query.where(Violation.severity == severity)
        if status: query = query.where(Violation.status == status)
        if camera_id: query = query.where(Violation.camera_id == camera_id)
        
        total = len(session.exec(query).all())
        res = session.exec(query.offset((page-1)*limit).limit(limit)).all()
        
        return {"data": res, "meta": {"page": page, "limit": limit, "total_data": total, "total_pages": math.ceil(total/limit)}}

@app.patch("/api/violations/{violation_id}")
def update_status(violation_id: int, status: str, notes: str = None):
    with Session(engine) as session:
        v = session.get(Violation, violation_id)
        if v: v.status = status; v.admin_notes = notes; session.add(v); session.commit()
        return {"ok": True}

@app.delete("/api/clear")
def clear_db():
    with Session(engine) as session: session.exec(text("DELETE FROM violation")); session.commit()
    return {"status": "cleared"}

@app.get("/api/cameras")
def get_cameras():
    with Session(engine) as session: return session.exec(select(Camera)).all()