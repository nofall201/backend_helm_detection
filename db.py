from typing import Optional
from sqlmodel import Field, SQLModel, create_engine
from datetime import datetime
from enum import Enum

# --- ENUM STATUS (Workflow Satpam) ---
class ViolationStatus(str, Enum):
    PENDING = "PENDING"       # Baru masuk, belum dicek
    CONFIRMED = "CONFIRMED"   # Valid (Benar pelanggaran)
    REJECTED = "REJECTED"     # Invalid (Salah deteksi/False Positive)

# --- TABEL 1: KAMERA (DAFTAR PERANGKAT IOT) ---
class Camera(SQLModel, table=True):
    id: str = Field(primary_key=True)       # ID Unik (misal: "ID-GC609")
    location: str                           # Nama Lokasi (misal: "Gudang Belakang")
    is_active: bool = Field(default=True)   # Status aktif/mati
    created_at: datetime = Field(default_factory=datetime.now)

# --- TABEL 2: PELANGGARAN (DATA DETEKSI) ---
class Violation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.now, index=True)
    
    # Data Hasil Deteksi YOLO
    person_count: int
    helmet_count: int
    screenshot_path: str
    confidence: float       # Skor keyakinan AI (0.0 - 1.0)
    severity: str           # "CRITICAL" atau "WARNING"
    
    # Relasi ke Kamera (Foreign Key)
    # Menyimpan ID kamera mana yang menangkap pelanggaran ini
    camera_id: str = Field(foreign_key="camera.id") 
    
    # Data Audit / Penanganan
    status: ViolationStatus = Field(default=ViolationStatus.PENDING)
    admin_notes: Optional[str] = None # Catatan manual satpam

# --- KONFIGURASI DATABASE SQLITE ---
sqlite_file_name = "violations.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

# connect_args={"check_same_thread": False} diperlukan untuk SQLite di FastAPI
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)