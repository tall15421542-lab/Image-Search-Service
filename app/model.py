import uuid
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import Uuid
from sqlalchemy.sql import func

Base = declarative_base()

class Inference(Base):
    __tablename__ = "inferences"
    id = Column(Uuid, primary_key=True, index=True, default=uuid.uuid4)
    model_name = Column(String, nullable=False)
    input_text = Column(String, nullable=False)
    output_image = Column(String, nullable=False)
    feedback = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())

