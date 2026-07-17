from pydantic import BaseModel, Field
from typing import Optional

# 1. Someone needs blood
class EmergencyBloodRequest(BaseModel):
    patient_name: str
    blood_group: str = Field(..., description="Must be a valid blood type like A+, O-")
    units_required: int = Field(..., gt=0, description="Must be at least 1 unit")
    hospital_name: str
    emergency_notes: Optional[str] = None
    latitude: float
    longitude: float

# 2. Someone is registering to give blood
class DonorProfile(BaseModel):
    full_name: str
    blood_group: str = Field(..., description="Must be a valid blood type like A+, O-")
    phone_number: str
    is_available_for_emergency: bool = True
    latitude: float
    longitude: float

# 3. A hospital in the network
class Hospital(BaseModel):
    hospital_name: str
    address: str
    emergency_contact: str
    has_active_trauma_center: bool = True