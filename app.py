"""
app.py
------
FastAPI REST service exposing the Churn prediction model.

Endpoints
---------
GET  /          – Health check
GET  /info      – Model info
POST /predict   – Predict churn for one customer
POST /retrain   – Retrain the model (perspective)

Start
-----
uvicorn app:app --reload --host 0.0.0.0 --port 8000

Docs
----
http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_pipeline import load_model, prepare_data, train_model, save_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "classifier.joblib"
DATA_PATH  = "Churn_Modelling.csv"

# Global model reference (loaded once at startup)
model = None


# ---------------------------------------------------------------------------
# Lifespan – load model at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model when the server starts."""
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"⚠️  No model found at {MODEL_PATH}. Train first via /retrain.")
    yield
    print("🛑 Server shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description="MLOps – Atelier 4 : expose the predict() function as a REST service.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """Input features for one customer."""
    CreditScore:       float = Field(..., example=619,       description="Credit score")
    Gender:            int   = Field(..., example=0,         description="0 = Female, 1 = Male")
    Age:               float = Field(..., example=42,        description="Customer age")
    Tenure:            float = Field(..., example=2,         description="Years with bank")
    Balance:           float = Field(..., example=0.0,       description="Account balance")
    NumOfProducts:     int   = Field(..., example=1,         description="Number of products")
    HasCrCard:         int   = Field(..., example=1,         description="Has credit card (0/1)")
    IsActiveMember:    int   = Field(..., example=1,         description="Is active member (0/1)")
    EstimatedSalary:   float = Field(..., example=101348.88, description="Estimated salary")


class PredictionResponse(BaseModel):
    prediction:    int   # 0 = Not Exited, 1 = Exited
    label:         str   # "Exited" or "Not Exited"
    probability:   float # probability of churn


class RetrainRequest(BaseModel):
    n_estimators: Optional[int] = Field(100, description="Number of trees")
    test_size:    Optional[float] = Field(0.2, description="Test split ratio")


class RetrainResponse(BaseModel):
    message:  str
    accuracy: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    """Health check — confirms the API is running."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "message": "Churn Prediction API is running.",
    }


@app.get("/info", tags=["Health"])
def model_info():
    """Return basic information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /retrain first.")
    return {
        "model_type":    type(model).__name__,
        "n_estimators":  model.n_estimators,
        "model_path":    MODEL_PATH,
        "features": [
            "CreditScore", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
        ],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict whether a customer will churn.

    Send a JSON body with the 9 customer features.
    Returns prediction (0/1), label, and churn probability.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /retrain first.")

    try:
        features = [[
            customer.CreditScore,
            customer.Gender,
            customer.Age,
            customer.Tenure,
            customer.Balance,
            customer.NumOfProducts,
            customer.HasCrCard,
            customer.IsActiveMember,
            customer.EstimatedSalary,
        ]]

        prediction  = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])  # prob of churn
        label       = "Exited" if prediction == 1 else "Not Exited"

        return PredictionResponse(
            prediction=prediction,
            label=label,
            probability=round(probability, 4),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/retrain", response_model=RetrainResponse, tags=["Training"])
def retrain(params: RetrainRequest = RetrainRequest()):
    """
    Retrain the model with fresh data and optional hyperparameters.

    Returns accuracy on the test set after retraining.
    """
    global model
    try:
        from sklearn.metrics import accuracy_score

        x_train, x_test, y_train, y_test = prepare_data(
            DATA_PATH, test_size=params.test_size
        )
        model = train_model(x_train, y_train, n_estimators=params.n_estimators)
        save_model(model, MODEL_PATH)

        y_pred   = model.predict(x_test)
        accuracy = round(float(accuracy_score(y_test, y_pred)), 4)

        return RetrainResponse(
            message=f"Model retrained with {params.n_estimators} estimators and saved.",
            accuracy=accuracy,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")
