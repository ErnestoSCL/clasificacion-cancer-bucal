from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

try:
    from .database import Base
except ImportError:
    from database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    project = Column(String(100))
    image_name = Column(String(255))
    predicted_label = Column(String(50))
    confidence = Column(Float)
    probabilities = Column(JSON)
    is_correct = Column(Boolean, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
