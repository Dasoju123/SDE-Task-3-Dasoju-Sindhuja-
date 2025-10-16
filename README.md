# SDE-Task-3-Dasoju-Sindhuja-
# HealthSearch - Minimal FastAPI Implementation



## Overview

This project implements a small FastAPI service that:

- Accepts clinical notes (`POST /add_note`)

- Generates embeddings (real model if installed; fallback deterministic hash)

- Stores notes in-memory (easy to swap for DB)

- Performs semantic search using cosine similarity (`GET /search_notes`)

- Basic token-based authentication



## Files

- `main.py` - FastAPI app

- `models.py` - Pydantic models and helper embedding function

- `requirements.txt`



## Setup (local)

1. Create a Python virtual environment and activate it:

   ```bash

   python -m venv .venv

   source .venv/bin/activate   # Linux/Mac

   .venv\Scripts\activate      # Windows
