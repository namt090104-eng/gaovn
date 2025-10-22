import streamlit as st
import pickle
import numpy as np
import pandas as pd 

# ==========================================================
# CẤU HÌNH CỐT LÕI (CẦN THAY ĐỔI THEO DỰ ÁN CỦA BẠN)
# ==========================================================
# Ghi chú: Các file .pkl phải được đặt cùng thư mục với file app.py này
MODEL_PATH = 'random_forest_model.pkl'  # Tên file mô hình RF đã lưu
SCALER_PATH = 'scaler.pkl'            # Tên file StandardScaler đã lưu
PCA_PATH = 'pca.pkl'                  # Tên file PCA đã lưu

# Tên các lớp gạo (Cần khớp chính xác với thứ tự khi huấn luyện mô hình)
CLASS_NAMES = [
    'Arborio (Gạo Tròn)', 
    'Basmati (Gạo Thon Dài)', 
    'Ipsala', 
    'Jasmine (Gạo Thơm)', 
    'Karacadag'
]

# Tên các đặc trưng gốc (PHẢI KHỚP VỚI ĐẶC TRƯNG ĐẦU VÀO CỦA SCALER/PCA)
FEATURE_NAMES = [
    'Area (Diện tích)', 
    'Perimeter (Chu vi)', 
    'MajorAxisLength (Chiều dài trục chính)', 
    'MinorAxisLength (Chiều dài trục phụ)',    
    'Eccentricity (Độ lệch tâm)', 
    'ConvexArea (Diện tích lồi)', 
    'EquivalentDiameter (Đường kính tương đương)'
]

# Giá trị mặc định cho ô nhập liệu (Có thể lấy trung bình của tập dữ liệu)
DEFAULT_FEATURE_VALUES = [7000, 350, 100, 80, 0.7, 7050, 95] 

# Giá trị tối đa giả định cho thanh trượt
MAX_FEATURE_VALUES = [15000, 800, 250, 150, 1.0, 15500, 180]
# ==========================================================


@st.cache_resource
def load_rf_components():
    """Tải mô hình Random Forest, Scaler và PCA từ file .pkl."""
    st.info("Đang tải các thành phần mô hình (Model, Scaler, PCA)...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(PCA_PATH, 'rb') as f:
            pca = pickle.load(f)
        st.success("Tải mô hình thành công!")
        return model, scaler, pca
    except FileNotFoundError as e:
        st.error(f"Lỗi tải file: Vui lòng kiểm tra các file .pkl ({e.filename}) đã được đặt cùng thư mục với file Python này chưa.")
        # Dừng ứng dụng nếu không tìm thấy file
        st.stop() 
    except Exception as e:
        st.error(f"Lỗi tải hoặc giải nén mô hình: {e}")
        st.stop()

def predict_features(input_data, model, scaler, pca, class_names):
    """Tiền xử lý dữ liệu đặc trưng và dự đoán."""
    try:
        # 1. Chuyển đổi thành mảng numpy 2D (1 mẫu, N đặc trưng)
        features = np.array(input_data).reshape(1, -1)
        
        # 2. Tiền xử lý (Phải theo đúng thứ tự: Scaling -> PCA)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # 3. Dự đoán xác suất
        predictions = model.predict_proba(features_pca)[0] # Lấy mảng xác suất 1D
        
        # Lấy top 3 dự đoán
        top_k_indices = np.argsort(predictions)[::-1]
        
        # Tạo DataFrame cho kết quả chi tiết
        results = pd.DataFrame({
            'Loại Gạo': [class_names[i] for i in top_k_indices],
            'Xác suất': [predictions[i] for i in top_k_indices]
        })
        results['Xác suất'] = results['Xác suất'].apply(lambda x: f"{x:.2%}")
        
        predicted_class_index = top_k_indices[0]
        confidence = predictions[predicted_class_index]
        
        return class_names[predicted_class_index], confidence, results

    except ValueError as e:
        st.error(f"Lỗi định dạng dữ liệu: Số lượng đặc trưng đầu vào không khớp với mô hình. {e}")
        return "Lỗi xử lý", 0.0, pd.DataFrame()
    except Exception as e:
        st.error(f"Lỗi trong quá trình dự đoán (PCA/Model): {e}")
        return "Lỗi xử lý", 0.0, pd.DataFrame()


# ==========================================================
# LOGIC ỨNG DỤNG CHÍNH
# ==========================================================
model, scaler, pca = load_rf_components()

st.set_page_config(page_title="Phân Loại Gạo Random Forest", layout="wide")

st.title("🌾 Ứng Dụng Phân Biệt Loại Gạo (Random Forest)")
st.markdown(f"**Mô hình:** Random Forest | **Đặc trưng:** {len(FEATURE_NAMES)} đặc trưng hình thái ({pca.n_components_} thành phần chính)")
st.markdown("---")


# -----------------
# 1. Giao diện Nhập liệu (Sidebar)
# -----------------
with st.sidebar:
    st.header("1. Nhập Đặc trưng Hạt Gạo")
    st.info("Sử dụng các thanh trượt bên dưới để nhập các giá trị đo được của hạt gạo mới.")
    
    input_data = []
    
    # Tạo ô nhập liệu cho từng đặc trưng
    for i, feature in enumerate(FEATURE_NAMES):
        # Thiết lập giới hạn min/max dựa trên cấu hình
        min_val = 0.0
        max_val = MAX_FEATURE_VALUES[i] if i < len(MAX_FEATURE_VALUES) else 1000
        default_val = DEFAULT_FEATURE_VALUES[i] if i < len(DEFAULT_FEATURE_VALUES) else 0

        val = st.slider(
            f"**{i+1}. {feature}**", 
            min_value=min_val, 
            max_value=max_val,
            value=default_val,
            step=0.01 if max_val <= 1.0 else 1.0,
            format="%.2f"
        )
        input_data.append(val)

# -----------------
# 2. Nút Dự đoán và Hiển thị Kết quả
# -----------------
st.subheader("2. Kết quả Phân loại:")

if st.sidebar.button('🚀 Phân Loại Hạt Gạo', type="primary", use_container_width=True):
    
    if len(input_data) != len(FEATURE_NAMES):
        st.error(f"Lỗi: Số lượng đặc trưng đầu vào ({len(input_data)}) không khớp với số lượng đặc trưng mô hình yêu cầu ({len(FEATURE_NAMES)}).")
    else:
        with st.spinner('Đang tiền xử lý dữ liệu và dự đoán...'):
            
            predicted_name, confidence, results_df = predict_features(
                input_data, model, scaler, pca, CLASS_NAMES
            )
            
            if predicted_name != "Lỗi xử lý":
                
                # Chia cột hiển thị kết quả chính
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(
                        label="Loại Gạo được dự đoán", 
                        value=predicted_name, 
                        delta=f"Độ tin cậy: {confidence:.2%}"
                    )
                    
                with col2:
                    st.info(f"Kết quả này được dự đoán với độ tin cậy **{confidence:.2%}** là loại **{predicted_name}**.")
                    
                st.markdown("---")
                
                # HIỂN THỊ CÁC KẾT QUẢ XÁC SUẤT CAO KHÁC
                st.subheader("Phân tích Xác suất Chi tiết:")
                
                # Hiển thị biểu đồ cột cho xác suất
                st.bar_chart(results_df.set_index('Loại Gạo')['Xác suất'].str.replace('%', '').astype(float))
                
                # Hiển thị bảng chi tiết
                st.dataframe(results_df, hide_index=True, use_container_width=True)
                
st.info("💡 Hướng dẫn: Nhập các đặc trưng ở thanh bên trái và nhấn nút 'Phân Loại Hạt Gạo' để xem kết quả dự đoán.")
