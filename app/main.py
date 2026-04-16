from fastapi import FastAPI
from app.api.endpoints import predict
from app.core.config import API_TITLE, API_VERSION

app = FastAPI(title=API_TITLE, version=API_VERSION)

# Inclure les routes
app.include_router(predict.router, prefix="/api/v1", tags=["prediction"])

@app.get("/")
def root():
    return {
        "message": f"{API_TITLE} is running",
        "version": API_VERSION,
        "endpoints": ["/api/v1/predict", "/api/v1/health"]
    }