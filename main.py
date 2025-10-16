# main.py

from fastapi import FastAPI, HTTPException, Depends, Header, status, Query

from fastapi.responses import JSONResponse

from typing import List, Optional

from models import AddNoteRequest, NoteRecord, SearchResultItem, deterministic_hash_embedding

import numpy as np

import time

import os

import logging



app = FastAPI(title="HealthSearch - FastAPI Demo")



# Simple in-memory store: list of dicts

STORE = []

NEXT_ID = 1



# Load API token from env or fallback

API_TOKEN = os.getenv("HEALTHSEARCH_TOKEN", "s3cr3t-t0ken")  # change for production



# Attempt to load a sentence-transformers model for real embeddings

USE_ST_MODEL = False

try:

    from sentence_transformers import SentenceTransformer

    # model choice can be changed. all-MiniLM-L6-v2 is common and compact

    model = SentenceTransformer("all-MiniLM-L6-v2")

    USE_ST_MODEL = True

    logging.info("SentenceTransformer model loaded. Using real embeddings.")

except Exception as e:

    model = None

    logging.info("sentence-transformers not available, using deterministic fallback embeddings.")



def get_embedding(text: str) -> np.ndarray:

    """

    Returns a unit-normalized embedding vector as numpy array.

    Uses sentence-transformers if available, otherwise deterministic fallback.

    """

    if USE_ST_MODEL:

        vec = model.encode([text], convert_to_numpy=True)[0]

        # normalize

        norm = np.linalg.norm(vec)

        if norm == 0:

            return vec

        return vec / norm

    else:

        # dimension match common models for parity in size

        return deterministic_hash_embedding(text, dim=384)



def require_token(authorization: Optional[str] = Header(None)):

    """

    Expect header: Authorization: Bearer <token>

    or X-API-KEY can be added if you prefer.

    """

    if authorization is None:

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    parts = authorization.split()

    if len(parts) == 2 and parts[0].lower() == "bearer":

        token = parts[1]

    else:

        token = authorization  # fallback: allow raw token

    if token != API_TOKEN:

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return True



@app.post("/add_note", status_code=201)

def add_note(payload: AddNoteRequest, auth: bool = Depends(require_token)):

    global NEXT_ID

    emb = get_embedding(payload.note)

    rec = {

        "id": NEXT_ID,

        "patient_id": payload.patient_id,

        "note": payload.note,

        "embedding": emb.tolist(),

        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    }

    STORE.append(rec)

    NEXT_ID += 1

    return JSONResponse(status_code=201, content={"id": rec["id"], "message": "Note added successfully."})



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:

    # both are assumed normalized; but be defensive

    na = np.linalg.norm(a)

    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:

        return 0.0

    return float(np.dot(a, b) / (na * nb))



@app.get("/search_notes", response_model=List[SearchResultItem])

def search_notes(query: str = Query(..., min_length=1), top_k: int = Query(3, ge=1, le=20),

                 auth: bool = Depends(require_token)):

    if len(STORE) == 0:

        return []

    q_emb = get_embedding(query)

    results = []

    for rec in STORE:

        emb = np.array(rec["embedding"], dtype=np.float64)

        score = cosine_similarity(q_emb, emb)

        results.append({

            "patient_id": rec["patient_id"],

            "note": rec["note"],

            "score": score

        })

    # sort descending by score

    results.sort(key=lambda x: x["score"], reverse=True)

    topn = results[:top_k]

    # format to SearchResultItem

    return [SearchResultItem(patient_id=r["patient_id"], note=r["note"], score=round(r["score"], 6)) for r in topn]



@app.get("/health")

def health():

    return {"status": "ok", "notes_count": len(STORE), "using_real_embeddings": USE_ST_MODEL}
