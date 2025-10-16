# models.py

from pydantic import BaseModel, Field

from typing import List, Optional

import hashlib

import numpy as np

import os



class AddNoteRequest(BaseModel):

    patient_id: str = Field(..., example="P001")

    note: str = Field(..., example="Patient reports chest pain and shortness of breath.")



class NoteRecord(BaseModel):

    id: int

    patient_id: str

    note: str

    embedding: List[float]

    created_at: Optional[str] = None



class SearchResultItem(BaseModel):

    patient_id: str

    note: str

    score: float



# deterministic fallback embedding generator (no external libs)

def deterministic_hash_embedding(text: str, dim: int = 384) -> np.ndarray:

    """

    Create a deterministic pseudo-embedding from a text by hashing.

    Not semantically meaningful, but deterministic and reproducible.

    """

    h = hashlib.sha256(text.encode('utf-8')).digest()

    # Expand bytes to required dimension by repeating hash bytes and interpreting as ints

    # produce floats in range [-1, 1] then normalize

    b = bytearray()

    while len(b) < dim * 4:

        h = hashlib.sha256(h).digest()

        b.extend(h)

    arr = np.frombuffer(bytes(b[:dim*4]), dtype=np.uint32).astype(np.float64)

    # scale down

    arr = (arr % 1000000) / 1000000.0  # [0,1)

    arr = (arr - 0.5) * 2.0  # [-1,1)

    # normalize to unit vector

    norm = np.linalg.norm(arr)

    if norm == 0:

        return arr

    return arr / norm
