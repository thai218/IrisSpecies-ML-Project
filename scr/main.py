from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import pandas as pd

# Load mô hình

model = joblib.load(r"models\best_model.pkl")  #RF model
scaler = joblib.load(r"models\scaler.pkl")
encoder = joblib.load(r"models\label_encoder.pkl")
scaler_area = joblib.load(r"models\scaler_area.pkl")

# ✅ In ra số lượng feature yêu cầu
print("⚠️ Scaler expects", scaler.scale_.shape[0], "features")


# Khởi tạo FastAPI
app = FastAPI(
    title="🌼 Iris Classifier API",
    description="Nhập các thông số cánh & đài hoa, API sẽ dự đoán loài hoa (Setosa, Versicolor, Virginica).",
    version="1.0"
)

# Kiểm tra API đang chạy
@app.get("/health")
def health_check():
    return {"status": "✅ API is active and running!"}

from fastapi.middleware.cors import CORSMiddleware

# CORS middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema đầu vào
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10)
    sepal_width:  float = Field(..., gt=0, le=10)
    petal_length: float = Field(..., gt=0, le=10)
    petal_width:  float = Field(..., gt=0, le=10)

    # field_validator với mode='before' để làm tròn ngay trước validate
    @field_validator("*", mode="before")
    def round_two_decimals(cls, v):
        # Làm tròn 2 chữ số thập phân cho nhất quán
        return round(v, 2)


# Endpoint chính
@app.post("/predict")
def predict_species(data: IrisInput):
    try:
        # 1. Mảng 4 feature gốc
        X_input = pd.DataFrame([{
            "sepal_length": data.sepal_length,
            "sepal_width": data.sepal_width,
            "petal_length": data.petal_length,
            "petal_width": data.petal_width
        }])
    
        # 2. Scale 4 feature gốc
        X_scaled = scaler.transform(X_input)

        # 3. Tính 2 feature area
        sepal_area = data.sepal_length * data.sepal_width
        petal_area = data.petal_length  * data.petal_width
        X_area = np.array([[sepal_area, petal_area]])

        # 4. Scale 2 feature area
        X_area_scaled = scaler_area.transform(X_area)

        # 5. Ghép lại thành (1,6)
        X_final = np.hstack([X_scaled, X_area_scaled])

        # 6. Dự đoán nhãn
        y_pred = model.predict(X_final)

        # 6.1. Dự đoán xác suất (chỉ dùng nếu model hỗ trợ)
        y_proba = model.predict_proba(X_final)
        confidence = round(np.max(y_proba) * 100, 2)  # Lấy xác suất cao nhất

        # 7. Giải mã nhãn
        species = encoder.inverse_transform(y_pred)[0]

        return {
            "prediction": species,
            "confidence": confidence  # trả thêm phần trăm
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))