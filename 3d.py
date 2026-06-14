import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import google.generativeai as genai
import math
import io

# =========================================================================
# 🔐 API 키 설정 구역 (AI 리포트용)
# =========================================================================
API_KEYS = {
    "GEMINI_API": st.secrets.get("GEMINI_API", ""),
}

if API_KEYS["GEMINI_API"]:
    genai.configure(api_key=API_KEYS["GEMINI_API"])

# =========================================================================
# 🎨 Streamlit 기본 UI 숨기기 및 설정
# =========================================================================
st.set_page_config(layout="wide", page_title="Micro-CT 콘크리트 미세구조 분석 AI V1.0")

hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden; position: relative;}
    header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# =========================================================================
# 🛠️ 유틸리티 함수
# =========================================================================
def generate_gemini_commentary(data_summary):
    if not API_KEYS["GEMINI_API"]:
        return """
[시스템 기본 분석 결과]
제공된 Micro-CT 슬라이스 데이터 분석 결과, 입력된 임계값(Threshold)에 따라 공극, 시멘트 페이스트, 골재의 부피비가 성공적으로 분리되었습니다. 분석된 체적비(Volume Fraction)와 혼합물 규칙(Rule of Mixtures)에 따라 도출된 체적 밀도(Bulk Density) 및 단위중량은 실제 배합 설계 및 구조적 내구성 평가를 위한 기초 자료로 활용될 수 있습니다. 특히 공극률 수치는 구조물의 투수성 및 열화 저항성을 간접적으로 지시하는 핵심 지표입니다.
*(Gemini API 미설정으로 인한 기본 코멘트입니다.)*
"""
    prompt = f"""
당신은 콘크리트 미세구조 및 X-ray CT 영상 분석 전문가입니다.
아래 분석 데이터를 바탕으로 학위 논문 또는 연구 보고서에 들어갈 전문적인 해석(5~7문장)을 작성하세요.
- 공극률이 내구성 및 강도에 미치는 영향 언급
- 단위중량 및 밀도 결과의 타당성 평가
- 혼합물 규칙(Rule of Mixtures) 적용 관점 반영

현장 데이터:
{data_summary}
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip() + "\n\n*(해당 분석 코멘트는 Gemini AI를 통해 생성되었습니다.)*"
    except Exception:
        return "AI 코멘트 생성 중 에러가 발생했습니다."

# =========================================================================
# ⚙️ 메인 프로그램 시작
# =========================================================================
st.title("🔬 3D Micro-CT 기반 콘크리트/모르타르 정밀 밀도 및 체적 분석 시스템")
st.markdown("> **X-ray 영상의 명암비(Grayscale) 감쇠 특성을 활용하여 공극, 페이스트, 골재의 3상(3-Phase) 체적비를 추출하고 혼합물 밀도 지배방정식을 연산합니다.**")

col_set1, col_set2 = st.columns([1, 2])

# -------------------------------------------------------------------------
# 1. 재료 밀도 기본 설정 구역
# -------------------------------------------------------------------------
with col_set1:
    st.subheader("📋 1. 구성 재료별 고유 밀도 설정")
    st.caption("실제 배합에 사용된 재료의 밀도를 입력하여 단위중량을 역산합니다.")
    
    rho_pore = 0.0 # 공극 밀도 (항상 0)
    st.number_input("내부 공극(Pore) 밀도 (g/cm³)", value=0.0, disabled=True)
    
    rho_paste = st.number_input("시멘트 페이스트 밀도 (g/cm³)", min_value=1.0, max_value=3.0, value=2.10, step=0.05)
    st.caption("※ 수화물 및 미세 겔구멍 포함 일반적 페이스트 기준")
    
    rho_agg = st.number_input("골재(Aggregate) 밀도 (g/cm³)", min_value=2.0, max_value=3.5, value=2.65, step=0.05)
    st.caption("※ 화강암 계열 굵은골재/잔골재 표건밀도 기준")

# -------------------------------------------------------------------------
# 2. 파일 업로드 구역
# -------------------------------------------------------------------------
with col_set2:
    st.subheader("📂 2. Micro-CT 단면(Slice) 이미지 업로드")
    st.warning("⚠️ 웹 환경에서는 메모리 제한이 있습니다. 1,500장 전체 대신 **대표 슬라이스 10~50장**을 추출하여 업로드하는 것을 권장합니다.")
    
    uploaded_files = st.file_uploader(
        "Micro-CT 영상 파일 선택 (.bmp, .png, .jpg 등 다중 선택 가능)", 
        type=["bmp", "png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

if not uploaded_files:
    st.info("파일을 업로드하면 분석이 시작됩니다.")
    st.stop()

# =========================================================================
# 📊 1단계: 대표 이미지 캘리브레이션 (Thresholding)
# =========================================================================
st.write("---")
st.subheader("📊 3. 이미지 명암비 분할(Segmentation) 임계값 설정")
st.markdown("X-ray 투과율에 따라 밀도가 낮은 **공극은 검은색(0)**, **페이스트는 회색**, 밀도가 높은 **골재는 흰색(255)** 부근으로 표현됩니다.")

# 첫 번째 이미지를 대표로 불러와서 히스토그램 및 프리뷰 제공
sample_file = uploaded_files[0]
sample_img = Image.open(sample_file).convert("L") # 흑백(Grayscale) 변환
img_arr = np.array(sample_img)

h, w = img_arr.shape
st.write(f"**대표 슬라이스 해상도:** `{w} x {h} pixels` (업로드된 총 {len(uploaded_files)}장 분석 대기 중)")

c_sl1, c_sl2 = st.columns(2)
with c_sl1:
    thresh_pore = st.slider("🔴 공극(Pore) 임계값 (0 ~ 값 이하)", min_value=10, max_value=100, value=45, step=1)
with c_sl2:
    thresh_agg = st.slider("🔵 골재(Aggregate) 임계값 (값 ~ 255 이상)", min_value=100, max_value=250, value=140, step=1)

# 히스토그램 그리기
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.hist(img_arr.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
ax.axvline(thresh_pore, color='red', linestyle='dashed', linewidth=2, label=f'Pore Thresh ({thresh_pore})')
ax.axvline(thresh_agg, color='blue', linestyle='dashed', linewidth=2, label=f'Agg Thresh ({thresh_agg})')
ax.legend()
ax.set_title("Grayscale Histogram of Sample Slice")
ax.set_xlabel("Pixel Intensity (0:Black -> 255:White)")
ax.set_ylabel("Frequency")

# 분할(Segmentation) 시각화 마스크 생성
color_mask = np.zeros((h, w, 3), dtype=np.uint8)
color_mask[img_arr <= thresh_pore] = [255, 0, 0]          # 공극: 빨간색
color_mask[(img_arr > thresh_pore) & (img_arr < thresh_agg)] = [200, 200, 200] # 페이스트: 연회색
color_mask[img_arr >= thresh_agg] = [0, 0, 255]          # 골재: 파란색

c_v1, c_v2, c_v3 = st.columns(3)
with c_v1:
    st.image(img_arr, caption="1. 원본 CT 이미지 (Grayscale)", use_container_width=True, clamp=True)
with c_v2:
    st.pyplot(fig)
    st.caption("2. 밝기 분포 및 사용자 임계값 지정")
with c_v3:
    st.image(color_mask, caption="3. 분할 시각화 (빨강:공극 / 파랑:골재)", use_container_width=True)

# =========================================================================
# 🚀 2단계: 전체 스택 3D 볼륨(Volume) 연산 처리
# =========================================================================
st.write("---")

if st.button("🚀 업로드된 모든 단면 분석 및 밀도 연산 시작", type="primary"):
    total_voxels = 0
    pore_voxels = 0
    paste_voxels = 0
    agg_voxels = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 메모리 오버플로우 방지를 위한 순차 처리 방식
    for idx, f in enumerate(uploaded_files):
        img_temp = Image.open(f).convert("L")
        arr_temp = np.array(img_temp)
        
        total_voxels += arr_temp.size
        pore_voxels += np.sum(arr_temp <= thresh_pore)
        agg_voxels += np.sum(arr_temp >= thresh_agg)
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
        status_text.text(f"분석 중... {idx + 1} / {len(uploaded_files)} 파일 처리 완료")
        
    paste_voxels = total_voxels - pore_voxels - agg_voxels
    
    # 부피비(Volume Fraction) 퍼센트 계산
    vol_pore = (pore_voxels / total_voxels) * 100.0
    vol_paste = (paste_voxels / total_voxels) * 100.0
    vol_agg = (agg_voxels / total_voxels) * 100.0
    
    # 혼합물 규칙(Rule of Mixtures)에 따른 밀도 및 단위중량 추정
    # 밀도(g/cm3) = Σ (체적비 * 고유밀도)
    bulk_density = (vol_pore/100.0 * rho_pore) + (vol_paste/100.0 * rho_paste) + (vol_agg/100.0 * rho_agg)
    unit_weight = bulk_density * 1000.0 # g/cm3 -> kg/m3 변환
    
    status_text.text("✅ 전체 슬라이스 3D 체적 분석 및 물리량 연산 완료!")
    
    # =========================================================================
    # 🧾 3단계: 분석 결과 리포트 출력
    # =========================================================================
    st.markdown("### 🏆 Micro-CT 3D 정량 분석 결과")
    
    st.info(f"💡 **분석 메타데이터:** 총 {len(uploaded_files)}장의 단면 이미지에서 **{total_voxels:,} 개**의 복셀(Voxel) 데이터를 처리했습니다.")
    
    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric(label="내부 공극률 (Porosity)", value=f"{vol_pore:.2f} %", delta="구조 취약 및 투수성 유발인자", delta_color="inverse")
    res_c2.metric(label="페이스트 체적비", value=f"{vol_paste:.2f} %", delta="수화 생성물", delta_color="off")
    res_c3.metric(label="골재(잔/굵은) 체적비", value=f"{vol_agg:.2f} %", delta="강도 발현 주재료", delta_color="normal")
    
    st.write("---")
    
    st.markdown("#### ⚖️ 체적 밀도(Bulk Density) 및 단위중량 산출")
    st.latex(r"\rho_{bulk} = (V_{pore} \times \rho_{pore}) + (V_{paste} \times \rho_{paste}) + (V_{agg} \times \rho_{agg})")
    
    d_c1, d_c2 = st.columns(2)
    d_c1.success(f"📌 **추정 체적 밀도 (Bulk Density):** `{bulk_density:.3f} g/cm³`")
    d_c2.error(f"📌 **추정 단위중량 (Unit Weight):** `{unit_weight:,.1f} kg/m³`")
    
    st.caption("※ 위 결과는 X-ray 감쇠 계수 기반 체적비와 사용자가 입력한 이론 밀도를 결합하여 추정한 결과이며, 실제 절대 건조 비중 및 수중 중량 측정 시험 결과와 대조하여 임계값(Threshold)을 미세 조정할 수 있습니다.")

    # =========================================================================
    # 🤖 Gemini AI 논문용 코멘트 생성
    # =========================================================================
    st.write("---")
    st.subheader("🤖 AI 기반 학술 분석 리포트")
    
    summary_text = f"분석 이미지 수: {len(uploaded_files)}장, 총 픽셀수: {total_voxels}개. 측정된 공극률은 {vol_pore:.2f}%, 시멘트 페이스트 체적비는 {vol_paste:.2f}%, 골재 체적비는 {vol_agg:.2f}% 임. 입력된 재료 밀도(페이스트 {rho_paste}g/cm3, 골재 {rho_agg}g/cm3)를 Rule of Mixtures에 대입한 결과, 추정된 체적 밀도는 {bulk_density:.3f} g/cm3, 단위중량은 {unit_weight:.1f} kg/m3 임."
    
    with st.spinner("Gemini AI가 학위 논문용 분석 소견을 작성 중입니다..."):
        ai_report = generate_gemini_commentary(summary_text)
        st.info(ai_report)
