from pydantic import BaseModel, Field
from typing import Optional

# This enforces that the API will reject any request that is missing these fields
class EmergencyBloodRequest(BaseModel):
    patient_name: str
    blood_group: str = Field(..., description="Must be a valid blood type like A+, O-")
    units_required: int = Field(..., gt=0, description="Must be at least 1 unit")
    hospital_name: str
    emergency_notes: Optional[str] = None
    latitude: float
    longitude: float