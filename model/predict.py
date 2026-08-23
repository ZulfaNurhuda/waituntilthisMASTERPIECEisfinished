
import joblib
import numpy as np
import pandas as pd

from features import extract_features, FEATURE_COLUMNS

MODEL_PATH = "shelf_life_model.joblib"
Q10_PATH = "shelf_life_model_q10.joblib"
Q90_PATH = "shelf_life_model_q90.joblib"

_model = None
_model_q10 = None
_model_q90 = None


def _load_models():
    global _model, _model_q10, _model_q90
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _model_q10 = joblib.load(Q10_PATH)
        _model_q90 = joblib.load(Q90_PATH)
    return _model, _model_q10, _model_q90


def _status_from_hours(hours: float) -> str:
    days = hours / 24.0
    if days >= 3:
        return "layak"
    elif days >= 1:
        return "waspada"
    else:
        return "segera_tindak_lanjuti"


def predict_remaining_shelf_life(riwayat_suhu: list, komoditas: str = "cabai_merah_giling") -> dict:
    if komoditas != "cabai_merah_giling":
        raise ValueError(
            f"MVP hanya mendukung komoditas 'cabai_merah_giling', dapat: {komoditas}"
        )

    temps = [r["suhu_celsius"] for r in riwayat_suhu]
    feats = extract_features(temps)
    X = pd.DataFrame([feats])[FEATURE_COLUMNS]

    model, model_q10, model_q90 = _load_models()
    point = float(model.predict(X)[0])
    q10 = float(model_q10.predict(X)[0])
    q90 = float(model_q90.predict(X)[0])
    q10, point, q90 = sorted([q10, point, q90]) 

    interval_width = q90 - q10

    relative_width = interval_width / max(point, 1e-6)
    confidence = float(np.clip(1.0 - relative_width / 4.0, 0.05, 0.99))

    return {
        "prediksi_sisa_umur_simpan_jam": round(point, 1),
        "status": _status_from_hours(point),
        "confidence": round(confidence, 2),
        "interval_jam": [round(q10, 1), round(q90, 1)],
    }


if __name__ == "__main__":
    demo_input = [
        {"timestamp": "2026-08-21T08:00:00Z", "suhu_celsius": 4.2},
        {"timestamp": "2026-08-21T09:00:00Z", "suhu_celsius": 5.1},
        {"timestamp": "2026-08-21T10:00:00Z", "suhu_celsius": 6.8},
        {"timestamp": "2026-08-21T11:00:00Z", "suhu_celsius": 12.5},
        {"timestamp": "2026-08-21T12:00:00Z", "suhu_celsius": 22.0},
        {"timestamp": "2026-08-21T13:00:00Z", "suhu_celsius": 28.5},
    ]
    result = predict_remaining_shelf_life(demo_input, komoditas="cabai_merah_giling")
    print(result)
