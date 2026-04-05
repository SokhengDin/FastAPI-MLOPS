from pydantic import BaseModel


class PredictRequest(BaseModel):
    Soil_Type                   : str
    Soil_pH                     : float
    Soil_Moisture               : float
    Organic_Carbon              : float
    Electrical_Conductivity     : float
    Temperature_C               : float
    Humidity                    : float
    Rainfall_mm                 : float
    Sunlight_Hours              : float
    Wind_Speed_kmh              : float
    Crop_Type                   : str
    Crop_Growth_Stage           : str
    Season                      : str
    Irrigation_Type             : str
    Water_Source                : str
    Field_Area_hectare          : float
    Mulching_Used               : str
    Previous_Irrigation_mm      : float
    Region                      : str


class ClassProbability(BaseModel):
    label       : str
    probability : float


class PredictResponse(BaseModel):
    irrigation_need : str
    confidence      : float
    probabilities   : list[ClassProbability]
