from fastapi import FastAPI
from .main import app
from .predict import router as predict_router
from .retrieve import router as retrieve_router
from .hybrid_retrieve import router as hybrid_retrieve_router

app.include_router(predict_router)
app.include_router(retrieve_router)
app.include_router(hybrid_retrieve_router)
