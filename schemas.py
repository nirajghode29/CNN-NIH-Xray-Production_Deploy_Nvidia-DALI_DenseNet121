from pydantic import BaseModel
from typing import List, Dict

class PredictionResponse(BaseModel):
    predicted_labels: List[str]
    probabilities: Dict[str, float]
    threshold: float