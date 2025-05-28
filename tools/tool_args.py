from pydantic import BaseModel, field_validator, ValidationError
from langchain.tools import BaseTool

class FlyToArgs(BaseModel):
    """Relative movement in metres (N, E, D)."""
    north: float
    east:  float
    down:  float

    # --- make the input forgiving ------------------------------------------
    @field_validator('north', 'east', 'down', mode='before')
    @classmethod
    def _coerce_number(cls, v):
        # Accept "5", 5, "5.0", etc.
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            v = v.strip()
            if v.replace(".", "", 1).replace("-", "", 1).isdigit():
                return float(v)
        raise ValueError("must be a number")

    @classmethod
    def model_validate(cls, obj, **kwargs):
        """
        Overload the top-level validator so we can accept:
          • "5,0,0"        ← string
          • "5 0 0"
          • [5, 0, 0]      ← list/tuple
        """
        if isinstance(obj, (list, tuple)) and len(obj) == 3:
            obj = {"north": obj[0], "east": obj[1], "down": obj[2]}
        elif isinstance(obj, str):
            parts = obj.replace(",", " ").split()
            if len(parts) == 3:
                obj = {"north": parts[0], "east": parts[1], "down": parts[2]}
        return super().model_validate(obj, **kwargs)