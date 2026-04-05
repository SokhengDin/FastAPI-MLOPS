from typing import Optional, TypeVar, Generic, List
from pydantic import BaseModel, Field


T = TypeVar('T')

class RESPONSE_SCHEMA(BaseModel, Generic[T]):
    status      : int
    message     : str
    data        : Optional[T] = None