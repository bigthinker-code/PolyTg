
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_service import predict_tg


app = FastAPI(
    title="PolyTg API",
    description=(
        "API for predicting polymer glass transition "
        "temperature using machine learning."
    ),
    version="1.0.0"
)


# Temporary CORS configuration for development.
# We will restrict this to your deployed frontend later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    psmiles: str = Field(
        ...,
        min_length=1,
        description="Polymer SMILES (PSMILES) structure"
    )


@app.get("/")
def home():
    return {
        "message": "PolyTg API is running.",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "PolyTg API"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        result = predict_tg(request.psmiles)
        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
