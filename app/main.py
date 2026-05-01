from fastapi import FastAPI
from app.api.endpoints import predict
from app.core.config import API_TITLE, API_VERSION
import os, psutil

app = FastAPI(title=API_TITLE, version=API_VERSION)
app.include_router(predict.router, prefix="/api/v1", tags=["prediction"])

@app.on_event("startup")
async def startup_event():
    mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"[STARTUP] Server up. RSS = {mb:.1f} MB. Model will load on first /predict request.")

@app.get("/")
def root():
    return {"message": f"{API_TITLE} is running", "version": API_VERSION}
