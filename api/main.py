import logging

from flask import Flask, jsonify, request

LOGGER = logging.getLogger("oralscan_api")

try:
    from .model import predict_oral_cancer
    from .utils import load_image_from_bytes
except ImportError:
    from model import predict_oral_cancer
    from utils import load_image_from_bytes

DB_ENABLED = True
engine = None
get_db = None
Base = None
Prediction = None

try:
    try:
        from .database import engine, get_db
        from .models_db import Base, Prediction
    except Exception:
        from database import engine, get_db
        from models_db import Base, Prediction
except Exception as exc:
    DB_ENABLED = False
    LOGGER.warning("Persistencia en DB deshabilitada: %s", exc)

if DB_ENABLED and Base is not None and engine is not None:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        DB_ENABLED = False
        LOGGER.warning("No se pudo inicializar la tabla de predicciones: %s", exc)


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/model-info")
def model_info():
    return jsonify(
        {
            "project": "app_clasificacion_cancer_bucal",
            "architecture": "MobileNetV2 + custom dense head",
            "classes": ["Normal", "Cancer"],
            "input_size": "224x224",
        }
    )


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Debe enviar un archivo en el campo 'file'"}), 400

    uploaded = request.files["file"]
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Archivo no válido"}), 400

    content_type = uploaded.content_type or ""
    if not content_type.startswith("image/"):
        return jsonify({"error": "El archivo debe ser una imagen válida"}), 400

    try:
        image_bytes = uploaded.read()
        image = load_image_from_bytes(image_bytes)
        result = predict_oral_cancer(image)

        if DB_ENABLED and get_db is not None and Prediction is not None:
            try:
                db = next(get_db())
                db_prediction = Prediction(
                    project="app_clasificacion_cancer_bucal",
                    image_name=str(uploaded.filename),
                    predicted_label=str(result.get("predicted_label", "Unknown")),
                    confidence=float(result.get("confidence", 0.0)),
                    probabilities=result,
                    is_correct=None,
                )
                db.add(db_prediction)
                db.commit()
                db.close()
            except Exception as db_exc:
                LOGGER.warning("No se pudo guardar predicción en DB: %s", db_exc)

        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Error de inferencia: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
