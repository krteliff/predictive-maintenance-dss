import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# --- SAYFA AYARI ---
st.set_page_config(
    page_title="Makine Arıza Karar Destek Sistemi",
    page_icon="🏭",
    layout="wide"
)

# --- MODELİ YÜKLE ---
@st.cache_resource
def model_yukle():
    df = pd.read_csv("/Users/elif/Downloads/ai4i2020.csv")
    df_model = df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])
    df_model["Type"] = df_model["Type"].map({"L": 0, "M": 1, "H": 2})
    df_majority = df_model[df_model["Machine failure"] == 0]
    df_minority = df_model[df_model["Machine failure"] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    X = df_balanced.drop(columns=["Machine failure"])
    y = df_balanced["Machine failure"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = model_yukle()

# --- BAŞLIK ---
st.title("🏭 Makine Arıza Karar Destek Sistemi")
st.markdown("Sensör değerlerini girin, sistem arıza riskini analiz etsin.")
st.divider()

# --- SENSORLER ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ Sensör Değerleri")
    machine_type = st.selectbox("Makine Tipi", ["L (Düşük)", "M (Orta)", "H (Yüksek)"])
    type_val = 0 if "L" in machine_type else 1 if "M" in machine_type else 2
    air_temp = st.slider("Hava Sıcaklığı (K)", 295.0, 305.0, 300.0, 0.1)
    process_temp = st.slider("Proses Sıcaklığı (K)", 305.0, 315.0, 310.0, 0.1)
    rot_speed = st.slider("Dönüş Hızı (rpm)", 1168, 2886, 1500)
    torque = st.slider("Tork (Nm)", 3.0, 77.0, 40.0, 0.5)
    tool_wear = st.slider("Takım Aşınması (dk)", 0, 253, 100)

with col2:
    st.subheader("🎯 Karar")

    sensor_verisi = pd.DataFrame([[type_val, air_temp, process_temp, rot_speed, torque, tool_wear]],
                                  columns=feature_names)
    olasilik = model.predict_proba(sensor_verisi)[0][1]

    if olasilik >= 0.75:
        st.error("🔴 KRİTİK ARIZA RİSKİ")
        st.markdown("### Hattı durdur, bakım ekibini çağır!")
        aksiyon = "Acil bakım planla, yedek parça hazırla"
    elif olasilik >= 0.40:
        st.warning("🟡 UYARI — Dikkat Gerekiyor")
        st.markdown("### 24 saat içinde kontrol et")
        aksiyon = "Bakım ekibini bilgilendir, izlemeyi artır"
    else:
        st.success("🟢 NORMAL — Üretim Devam Edebilir")
        st.markdown("### Sistem sağlıklı çalışıyor")
        aksiyon = "Rutin bakım takvimine devam et"

    st.metric("Arıza Olasılığı", f"%{olasilik*100:.1f}")
    st.info(f"📋 Önerilen Aksiyon: {aksiyon}")

    # Risk göstergesi
    st.progress(olasilik)

st.divider()

# --- GEÇMİŞ ANALİZ ---
st.subheader("📊 Veri Seti Özeti")
df = pd.read_csv("/Users/elif/Downloads/ai4i2020.csv")
col3, col4, col5 = st.columns(3)
with col3:
    st.metric("Toplam Kayıt", f"{len(df):,}")
with col4:
    st.metric("Toplam Arıza", f"{df['Machine failure'].sum():,}")
with col5:
    oran = round(df['Machine failure'].mean()*100, 1)
    st.metric("Arıza Oranı", f"%{oran}")

st.caption("Kaynak: AI4I 2020 Predictive Maintenance Dataset | Elif Kurt")