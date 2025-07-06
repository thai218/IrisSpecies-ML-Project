import streamlit as st
import requests

# Title
st.title("🌸 Dự đoán loài hoa Iris")
st.write("Nhập vào thông số hoa để dự đoán loài:")

# Input form
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0)
    sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0)

with col2:
    petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0)
    petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0)

# Gửi request đến FastAPI
if st.button("🔍 Dự đoán"):
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            confidence = result.get("confidence")  # an toàn hơn
            if confidence is not None:
                st.success(f"🌼 Loài hoa: **{result['prediction']}** ({confidence}%)")
            else:
                st.success(f"🌼 Loài hoa: **{result['prediction']}**")
        elif response.status_code == 422:
            errors = response.json()["detail"]
            st.error("⚠️ Lỗi nhập liệu:")
            for err in errors:
                loc = err.get("loc", ["?"])[-1]
                msg = err.get("msg", "")
                st.write(f"- `{loc}`: {msg}")        
        else:
            st.error("❌ Lỗi từ API: " + response.text)
    except Exception as e:
        st.error(f"Lỗi kết nối")
