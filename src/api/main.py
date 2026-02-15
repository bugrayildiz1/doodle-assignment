
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

from .predict import router as predict_router
from .retrieve import router as retrieve_router
from .hybrid_retrieve import router as hybrid_retrieve_router

from .analyze_anomalies import router as analyze_anomalies_router
from .feedback import router as feedback_router


app = FastAPI()

# Fallback mechanism: global exception handler
@app.exception_handler(Exception)
async def fallback_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}")
    # Fallback response (can be customized)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Service temporarily unavailable. Fallback activated.",
            "fallback_category": "Technical Issue"
        },
    )


app.include_router(predict_router)
app.include_router(retrieve_router)
app.include_router(hybrid_retrieve_router)
app.include_router(analyze_anomalies_router)
app.include_router(feedback_router)

@app.get("/")
def root():
    return {"message": "Support Ticket AI System is running."}
