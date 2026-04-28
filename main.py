import os
import uuid
import shutil
import traceback
from typing import Optional, Dict, Any

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from deepface import DeepFace

load_dotenv()

app = FastAPI(
    title="Presensi Face Verification Service",
    version="1.0.0"
)

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "change-me")

FACE_MODEL = os.getenv("FACE_MODEL", "Facenet512")
FACE_DETECTOR = os.getenv("FACE_DETECTOR", "opencv")
DISTANCE_METRIC = os.getenv("DISTANCE_METRIC", "cosine")

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "240"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "240"))

MIN_BLUR_SCORE = float(os.getenv("MIN_BLUR_SCORE", "35"))
MIN_BRIGHTNESS = float(os.getenv("MIN_BRIGHTNESS", "35"))
MAX_BRIGHTNESS = float(os.getenv("MAX_BRIGHTNESS", "225"))

ENABLE_DEEPFACE_ANTI_SPOOFING = os.getenv(
    "ENABLE_DEEPFACE_ANTI_SPOOFING", "false"
).lower() == "true"

TEMP_DIR = os.getenv("TEMP_DIR", "tmp")
os.makedirs(TEMP_DIR, exist_ok=True)


def check_token(x_verify_token: Optional[str]):
    if not VERIFY_TOKEN or VERIFY_TOKEN == "change-me":
        raise HTTPException(
            status_code=500,
            detail="VERIFY_TOKEN belum dikonfigurasi di verifier."
        )

    if x_verify_token != VERIFY_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def save_upload_file(upload_file: UploadFile, prefix: str) -> str:
    original_name = upload_file.filename or ""
    ext = os.path.splitext(original_name)[1].lower()

    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    file_name = f"{prefix}_{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(TEMP_DIR, file_name)

    size = 0

    with open(file_path, "wb") as buffer:
        while True:
            chunk = upload_file.file.read(1024 * 1024)
            if not chunk:
                break

            size += len(chunk)

            if size > MAX_FILE_SIZE_BYTES:
                buffer.close()
                if os.path.exists(file_path):
                    os.remove(file_path)

                raise HTTPException(
                    status_code=413,
                    detail=f"Ukuran file terlalu besar. Maksimal {MAX_FILE_SIZE_MB}MB."
                )

            buffer.write(chunk)

    return file_path


def read_image(path: str):
    image = cv2.imread(path)

    if image is None:
        return None

    return image


def image_quality_check(path: str) -> Dict[str, Any]:
    image = read_image(path)

    if image is None:
        return {
            "ok": False,
            "message": "File gambar tidak bisa dibaca.",
            "quality": {}
        }

    height, width = image.shape[:2]

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        return {
            "ok": False,
            "message": "Resolusi gambar terlalu kecil.",
            "quality": {
                "width": width,
                "height": height
            }
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    if blur_score < MIN_BLUR_SCORE:
        return {
            "ok": False,
            "message": "Foto terlalu blur.",
            "quality": {
                "width": width,
                "height": height,
                "blur_score": round(blur_score, 4),
                "brightness": round(brightness, 4)
            }
        }

    if brightness < MIN_BRIGHTNESS:
        return {
            "ok": False,
            "message": "Foto terlalu gelap.",
            "quality": {
                "width": width,
                "height": height,
                "blur_score": round(blur_score, 4),
                "brightness": round(brightness, 4)
            }
        }

    if brightness > MAX_BRIGHTNESS:
        return {
            "ok": False,
            "message": "Foto terlalu terang.",
            "quality": {
                "width": width,
                "height": height,
                "blur_score": round(blur_score, 4),
                "brightness": round(brightness, 4)
            }
        }

    return {
        "ok": True,
        "message": "Kualitas foto cukup.",
        "quality": {
            "width": width,
            "height": height,
            "blur_score": round(blur_score, 4),
            "brightness": round(brightness, 4)
        }
    }


def detect_single_face(path: str) -> Dict[str, Any]:
    try:
        faces = DeepFace.extract_faces(
            img_path=path,
            detector_backend=FACE_DETECTOR,
            enforce_detection=True,
            align=True
        )

        face_count = len(faces) if faces else 0

        if face_count != 1:
            return {
                "ok": False,
                "message": f"Jumlah wajah terdeteksi harus 1, terdeteksi {face_count}.",
                "face_count": face_count
            }

        return {
            "ok": True,
            "message": "Satu wajah terdeteksi.",
            "face_count": face_count
        }

    except Exception as e:
        return {
            "ok": False,
            "message": f"Wajah tidak terdeteksi: {str(e)}",
            "face_count": 0
        }


def calculate_score(distance: Optional[float], threshold: Optional[float]) -> Optional[float]:
    if distance is None or threshold is None or threshold <= 0:
        return None

    score = max(0, min(100, (1 - (distance / threshold)) * 100))
    return round(float(score), 4)


def run_deepface_verify(reference_path: str, selfie_path: str) -> Dict[str, Any]:
    base_kwargs = {
        "img1_path": reference_path,
        "img2_path": selfie_path,
        "model_name": FACE_MODEL,
        "detector_backend": FACE_DETECTOR,
        "distance_metric": DISTANCE_METRIC,
        "enforce_detection": True,
        "align": True,
    }

    if ENABLE_DEEPFACE_ANTI_SPOOFING:
        try:
            return DeepFace.verify(
                **base_kwargs,
                anti_spoofing=True
            )
        except TypeError:
            return DeepFace.verify(**base_kwargs)

    return DeepFace.verify(**base_kwargs)


def make_pending_response(message: str, extra: Optional[Dict[str, Any]] = None):
    return {
        "status": "pending_review",
        "verified": False,
        "score": None,
        "distance": None,
        "threshold": None,
        "method": "deepface",
        "message": message,
        "extra": extra or {}
    }


def make_rejected_response(message: str, extra: Optional[Dict[str, Any]] = None):
    return {
        "status": "rejected",
        "verified": False,
        "score": None,
        "distance": None,
        "threshold": None,
        "method": "deepface",
        "message": message,
        "extra": extra or {}
    }


async def verify_handler(
    reference_image: UploadFile,
    selfie_image: UploadFile,
    absensi_id: Optional[str],
    nik_karyawan: Optional[str],
    tanggal: Optional[str],
    presensi_challenge_id: Optional[str],
):
    reference_path = None
    selfie_path = None

    try:
        reference_path = save_upload_file(reference_image, "reference")
        selfie_path = save_upload_file(selfie_image, "selfie")

        reference_quality = image_quality_check(reference_path)
        if not reference_quality["ok"]:
            return make_pending_response(
                "Foto referensi tidak memenuhi standar kualitas.",
                {
                    "reference_quality": reference_quality,
                    "absensi_id": absensi_id,
                    "nik_karyawan": nik_karyawan,
                    "tanggal": tanggal,
                    "presensi_challenge_id": presensi_challenge_id,
                }
            )

        selfie_quality = image_quality_check(selfie_path)
        if not selfie_quality["ok"]:
            return make_rejected_response(
                selfie_quality["message"],
                {
                    "selfie_quality": selfie_quality,
                    "absensi_id": absensi_id,
                    "nik_karyawan": nik_karyawan,
                    "tanggal": tanggal,
                    "presensi_challenge_id": presensi_challenge_id,
                }
            )

        reference_face = detect_single_face(reference_path)
        if not reference_face["ok"]:
            return make_pending_response(
                "Foto referensi tidak valid.",
                {
                    "reference_face": reference_face,
                    "absensi_id": absensi_id,
                    "nik_karyawan": nik_karyawan,
                    "tanggal": tanggal,
                    "presensi_challenge_id": presensi_challenge_id,
                }
            )

        selfie_face = detect_single_face(selfie_path)
        if not selfie_face["ok"]:
            return make_rejected_response(
                selfie_face["message"],
                {
                    "selfie_face": selfie_face,
                    "absensi_id": absensi_id,
                    "nik_karyawan": nik_karyawan,
                    "tanggal": tanggal,
                    "presensi_challenge_id": presensi_challenge_id,
                }
            )

        result = run_deepface_verify(reference_path, selfie_path)

        verified = bool(result.get("verified", False))
        distance = result.get("distance")
        threshold = result.get("threshold")
        score = calculate_score(distance, threshold)

        status = "verified" if verified else "rejected"

        return {
            "status": status,
            "verified": verified,
            "score": score,
            "distance": round(float(distance), 6) if distance is not None else None,
            "threshold": round(float(threshold), 6) if threshold is not None else None,
            "method": "deepface",
            "message": "Wajah cocok." if verified else "Wajah tidak cocok.",
            "extra": {
                "model": FACE_MODEL,
                "detector": FACE_DETECTOR,
                "distance_metric": DISTANCE_METRIC,
                "anti_spoofing_enabled": ENABLE_DEEPFACE_ANTI_SPOOFING,
                "reference_quality": reference_quality,
                "selfie_quality": selfie_quality,
                "reference_face": reference_face,
                "selfie_face": selfie_face,
                "absensi_id": absensi_id,
                "nik_karyawan": nik_karyawan,
                "tanggal": tanggal,
                "presensi_challenge_id": presensi_challenge_id,
                "raw": result,
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        return make_pending_response(
            "Verifier error. Butuh review manual.",
            {
                "error": str(e),
                "trace": traceback.format_exc(),
                "absensi_id": absensi_id,
                "nik_karyawan": nik_karyawan,
                "tanggal": tanggal,
                "presensi_challenge_id": presensi_challenge_id,
            }
        )

    finally:
        for path in [reference_path, selfie_path]:
            if path and os.path.exists(path):
                os.remove(path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "presensi-face-verifier",
        "model": FACE_MODEL,
        "detector": FACE_DETECTOR,
        "distance_metric": DISTANCE_METRIC,
        "anti_spoofing_enabled": ENABLE_DEEPFACE_ANTI_SPOOFING,
    }


@app.post("/verify")
async def verify(
    reference_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...),
    absensi_id: Optional[str] = Form(None),
    nik_karyawan: Optional[str] = Form(None),
    tanggal: Optional[str] = Form(None),
    presensi_challenge_id: Optional[str] = Form(None),
    x_verify_token: Optional[str] = Header(None),
):
    check_token(x_verify_token)

    return await verify_handler(
        reference_image=reference_image,
        selfie_image=selfie_image,
        absensi_id=absensi_id,
        nik_karyawan=nik_karyawan,
        tanggal=tanggal,
        presensi_challenge_id=presensi_challenge_id,
    )


@app.post("/verify-face")
async def verify_face(
    reference_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...),
    absensi_id: Optional[str] = Form(None),
    nik_karyawan: Optional[str] = Form(None),
    tanggal: Optional[str] = Form(None),
    presensi_challenge_id: Optional[str] = Form(None),
    x_verify_token: Optional[str] = Header(None),
):
    check_token(x_verify_token)

    return await verify_handler(
        reference_image=reference_image,
        selfie_image=selfie_image,
        absensi_id=absensi_id,
        nik_karyawan=nik_karyawan,
        tanggal=tanggal,
        presensi_challenge_id=presensi_challenge_id,
    )