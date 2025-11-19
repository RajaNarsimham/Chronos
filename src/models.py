from sqlalchemy import Column, Integer, String, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base
import datetime

class Context(Base):
    __tablename__ = "contexts"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, index=True)
    context_id = Column(Integer, ForeignKey("contexts.id"))
    filename = Column(String)
    data = Column(Text)  # JSON string for timeseries
    min_value = Column(Float)
    max_value = Column(Float)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)

    context = relationship("Context")

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    context_id = Column(Integer, ForeignKey("contexts.id"))
    name = Column(String)
    path = Column(String)  # Path to saved model
    accuracy = Column(Float)
    trained_at = Column(DateTime, default=datetime.datetime.utcnow)

    context = relationship("Context")

class InferenceResult(Base):
    __tablename__ = "inference_results"
    id = Column(Integer, primary_key=True, index=True)
    metric_id = Column(Integer, ForeignKey("metrics.id"))
    model_id = Column(Integer, ForeignKey("models.id"))
    result = Column(Text)  # JSON string for predictions
    run_at = Column(DateTime, default=datetime.datetime.utcnow)

    metric = relationship("Metric")
    model = relationship("Model")