import os
import math
import cv2
import json
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# 에셋 실제 랜드마크 데이터베이스 로드
ASSET_LANDMARKS = {}
db_path = "primary_data/asset_landmarks.json"
if os.path.exists(db_path):
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            ASSET_LANDMARKS = json.load(f)
    except Exception as e:
        st.error(f"랜드마크 데이터베이스 파일 로드 실패: {e}")

# 황금 비율 캘리브레이션 랜드마크 스펙 정의 및 강제 보정 적용
CALIB_SPECS = {
    "muscles_torso.png": (90.0, 665.0, 142.0, 100.0),
    "nervous_torso.png": (95.0, 710.0, 146.0, 102.0),
    "skeleton_torso.png": (85.0, 670.0, 152.0, 106.0),
    "respiratory.png": (100.0, 710.0, 146.0, 102.0),
    "circulatory_torso.png": (75.0, 640.0, 152.0, 106.0),
    "digestive.png": (75.0, 610.0, 120.0, 70.0),
    "excretory.png": (20.0, 60.0, 140.0, 100.0)
}

for filename, spec in CALIB_SPECS.items():
    path = os.path.join("primary_data", filename)
    if os.path.exists(path):
        try:
            with Image.open(path) as img:
                w, h = img.size
            cx = w / 2.0
            sy, hy, s_r, h_r = spec
            ASSET_LANDMARKS[filename] = {
                "11": [float(cx + s_r), float(sy), 0.0],
                "12": [float(cx - s_r), float(sy), 0.0],
                "23": [float(cx + h_r), float(hy), 0.0],
                "24": [float(cx - h_r), float(hy), 0.0]
            }
        except Exception as e:
            pass

# 머리 에셋 전용 정밀 캘리브레이션 랜드마크 스펙 및 주입
CALIB_SPECS_HEAD = {
    "skeleton_head.png": (79.38, 181.21, 171.47, 135.05, 263.41),
    "nervous_head.png": (79.38, 181.21, 171.47, 135.05, 263.41),
    "muscles_head.png": (79.38, 181.21, 171.47, 135.05, 263.41),
    "circulatory_head.png": (79.38, 181.21, 171.47, 135.05, 263.41),
    "digestive_head.png": (79.38, 181.21, 171.47, 135.05, 263.41),
    "respiratory_head.png": (79.38, 181.21, 171.47, 135.05, 263.41)
}

for filename, spec in CALIB_SPECS_HEAD.items():
    path = os.path.join("primary_data", filename)
    if os.path.exists(path):
        try:
            r_eye_x, l_eye_x, eye_y, mouth_x, mouth_y = spec
            ASSET_LANDMARKS[filename] = {
                "3": [float(l_eye_x), float(eye_y), 0.0],
                "6": [float(r_eye_x), float(eye_y), 0.0],
                "9": [float(mouth_x - 10.0), float(mouth_y), 0.0],
                "10": [float(mouth_x + 10.0), float(mouth_y), 0.0]
            }
        except Exception as e:
            pass

# ==========================================
# 1. 고도화된 입체 장기 및 관절 마디 에셋 자동 생성
# ==========================================

def initialize_assets():
    """
    primary_data 폴더 내에 6대 신체 기관계 및 신경계, 관절 마디 에셋들을 입체적인 그래픽으로 자동 생성합니다.
    """
    os.makedirs("primary_data", exist_ok=True)
    
    # 생성할 마디 및 몸통 에셋 파일 정의
    assets = {
        "skeleton_head": "primary_data/skeleton_head.png",
        "skeleton_torso": "primary_data/skeleton_torso.png",
        "skeleton_upper_limb": "primary_data/skeleton_upper_limb.png",
        "skeleton_lower_limb": "primary_data/skeleton_lower_limb.png",
        "skeleton_upper_leg": "primary_data/skeleton_upper_leg.png",
        "skeleton_lower_leg": "primary_data/skeleton_lower_leg.png",
        
        "muscles_torso": "primary_data/muscles_torso.png",
        "muscles_upper_limb": "primary_data/muscles_upper_limb.png",
        "muscles_lower_limb": "primary_data/muscles_lower_limb.png",
        "muscles_upper_leg": "primary_data/muscles_upper_leg.png",
        "muscles_lower_leg": "primary_data/muscles_lower_leg.png",
        
        "nervous_head": "primary_data/nervous_head.png",
        "nervous_torso": "primary_data/nervous_torso.png",
        "nervous_upper_limb": "primary_data/nervous_upper_limb.png",
        "nervous_lower_limb": "primary_data/nervous_lower_limb.png",
        "nervous_upper_leg": "primary_data/nervous_upper_leg.png",
        "nervous_lower_leg": "primary_data/nervous_lower_leg.png",
        
        "circulatory_torso": "primary_data/circulatory_torso.png",
        "circulatory_upper_limb": "primary_data/circulatory_upper_limb.png",
        "circulatory_lower_limb": "primary_data/circulatory_lower_limb.png",
        "circulatory_upper_leg": "primary_data/circulatory_upper_leg.png",
        "circulatory_lower_leg": "primary_data/circulatory_lower_leg.png",
        
        "digestive": "primary_data/digestive.png",
        "respiratory": "primary_data/respiratory.png",
        "excretory": "primary_data/excretory.png",
        "guide": "primary_data/pose_guide.png"
    }

    # 하위 변수 매핑 지원 (기존 드로잉 코드와의 호환성 유지)
    skeleton_head_path = assets["skeleton_head"]
    skeleton_torso_path = assets["skeleton_torso"]
    skeleton_upper_limb_path = assets["skeleton_upper_limb"]
    skeleton_lower_limb_path = assets["skeleton_lower_limb"]
    skeleton_upper_leg_path = assets["skeleton_upper_leg"]
    skeleton_lower_leg_path = assets["skeleton_lower_leg"]
    
    muscles_torso_path = assets["muscles_torso"]
    muscles_upper_limb_path = assets["muscles_upper_limb"]
    muscles_lower_limb_path = assets["muscles_lower_limb"]
    muscles_upper_leg_path = assets["muscles_upper_leg"]
    muscles_lower_leg_path = assets["muscles_lower_leg"]
    
    nervous_head_path = assets["nervous_head"]
    nervous_torso_path = assets["nervous_torso"]
    nervous_upper_limb_path = assets["nervous_upper_limb"]
    nervous_lower_limb_path = assets["nervous_lower_limb"]
    nervous_upper_leg_path = assets["nervous_upper_leg"]
    nervous_lower_leg_path = assets["nervous_lower_leg"]
    
    circulatory_torso_path = assets["circulatory_torso"]
    circulatory_upper_limb_path = assets["circulatory_upper_limb"]
    circulatory_lower_limb_path = assets["circulatory_lower_limb"]
    circulatory_upper_leg_path = assets["circulatory_upper_leg"]
    circulatory_lower_leg_path = assets["circulatory_lower_leg"]
    
    digestive_path = assets["digestive"]
    respiratory_path = assets["respiratory"]
    excretory_path = assets["excretory"]
    
    # 1.0 머리 뼈대 (두개골, 200x200 크기)
    if not os.path.exists(skeleton_head_path):
        img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bone_color = (245, 245, 250, 220)
        shadow_color = (200, 200, 210, 220)
        
        # 두개골 동그라미 (중심 100, 85, 반지름 55)
        draw.ellipse([45, 30, 155, 140], fill=bone_color)
        draw.ellipse([50, 35, 150, 135], fill=shadow_color) # 입체 섀도우
        
        # 턱뼈
        draw.polygon([(65, 120), (135, 120), (120, 175), (80, 175)], fill=shadow_color)
        draw.polygon([(70, 125), (130, 125), (115, 170), (85, 170)], fill=bone_color)
        
        # 눈구멍 묘사 (검은 반투명 타원)
        draw.ellipse([65, 80, 95, 115], fill=(0, 0, 0, 80))
        draw.ellipse([105, 80, 135, 115], fill=(0, 0, 0, 80))
        
        # 콧구멍 묘사 (삼각형)
        draw.polygon([(95, 120), (105, 120), (100, 135)], fill=(0, 0, 0, 100))
        
        # 치아 구조 묘사 (약간의 빗금)
        for x in range(85, 120, 8):
            draw.line([x, 140, x, 150], fill=shadow_color, width=2)
            
        img.save(skeleton_head_path)

    # 1.1 몸통 뼈대
    if not os.path.exists(skeleton_torso_path):
        img = Image.new("RGBA", (800, 1000), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bone_color = (245, 245, 250, 220)
        shadow_color = (200, 200, 210, 220)
        
        # 척추뼈마디 배열 (입체 음영 부여)
        for y in range(260, 650, 30):
            draw.rounded_rectangle([380, y, 420, y + 22], radius=6, fill=shadow_color)
            draw.rounded_rectangle([385, y, 415, y + 20], radius=5, fill=bone_color)
            
        # 빗장뼈 및 어깨선
        draw.line([250, 250, 550, 250], fill=shadow_color, width=17)
        draw.line([250, 250, 550, 250], fill=bone_color, width=13)
        draw.ellipse([235, 240, 265, 270], fill=bone_color)
        draw.ellipse([535, 240, 565, 270], fill=bone_color)
        
        # 갈비뼈 (둥글고 입체적인 명암 적용)
        for y_offset in range(300, 480, 40):
            draw.arc([300, y_offset, 400, y_offset + 60], start=90, end=270, fill=shadow_color, width=10)
            draw.arc([300, y_offset, 400, y_offset + 60], start=90, end=270, fill=bone_color, width=7)
            draw.arc([400, y_offset, 500, y_offset + 60], start=270, end=90, fill=shadow_color, width=10)
            draw.arc([400, y_offset, 500, y_offset + 60], start=270, end=90, fill=bone_color, width=7)
            
        # 골반 (나비 모양 입체 묘사)
        draw.polygon([(280, 630), (520, 630), (480, 700), (320, 700)], fill=shadow_color)
        draw.polygon([(290, 635), (510, 635), (470, 695), (330, 695)], fill=bone_color)
        draw.ellipse([340, 645, 390, 685], fill=(0, 0, 0, 0))
        draw.ellipse([410, 645, 460, 685], fill=(0, 0, 0, 0))
        
        img.save(skeleton_torso_path)

    # 2.1 몸통 근육
    muscles_torso_path = "primary_data/muscles_torso.png"
    if not os.path.exists(muscles_torso_path):
        img = Image.new("RGBA", (800, 1000), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        muscle_color = (225, 90, 90, 205)
        line_color = (255, 200, 200, 120)
        # 대흉근
        draw.rounded_rectangle([270, 270, 390, 360], radius=15, fill=muscle_color)
        draw.rounded_rectangle([410, 270, 530, 360], radius=15, fill=muscle_color)
        # 복근
        for row in range(3):
            y_pos = 390 + row * 60
            draw.rounded_rectangle([330, y_pos, 390, y_pos + 50], radius=8, fill=muscle_color)
            draw.rounded_rectangle([410, y_pos, 470, y_pos + 50], radius=8, fill=muscle_color)
        # 삼각근
        draw.ellipse([210, 240, 270, 320], fill=muscle_color)
        draw.ellipse([530, 240, 590, 320], fill=muscle_color)
        img.save(muscles_torso_path)


    # 3.0 머리 신경계 (두뇌, 200x200 크기)
    if not os.path.exists(nervous_head_path):
        img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        brain_color = (180, 110, 240, 255)
        brain_light = (210, 150, 255, 255)
        
        # 대뇌 덩어리 묘사 (두개골 내경 공간 X:45~155, Y:30~140에 완전히 정밀 정합되도록 확장 및 조율)
        draw.ellipse([52, 35, 148, 115], fill=brain_color) # 중앙 대뇌
        for offset in [-14, 14]:
            draw.ellipse([52 + offset, 38, 148 + offset, 112], fill=brain_color) # 좌우 엽
        draw.ellipse([60, 40, 140, 100], fill=brain_light) # 대뇌 상단 밝은 영역
        
        # 소뇌 및 뇌간 (두개골 하단 및 목 시작 부위 매칭)
        draw.ellipse([72, 108, 128, 135], fill=(160, 90, 220, 255))
        draw.rounded_rectangle([93, 130, 107, 175], radius=3, fill=(140, 70, 200, 255))
        
        img.save(nervous_head_path)

    # 3.1 신경계 몸통 (nervous_torso.png)
    nervous_torso_path = "primary_data/nervous_torso.png"
    if not os.path.exists(nervous_torso_path):
        img = Image.new("RGBA", (800, 1000), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        brain_color = (180, 110, 240, 255)
        nerve_color = (200, 150, 255, 230)
        
        # 척수 신경망 (Y=260에서 요추 Y=650까지 세로 주선)
        draw.line([400, 260, 400, 650], fill=brain_color, width=12)
        draw.line([400, 260, 400, 650], fill=nerve_color, width=6)
        
        # 척수에서 갈비뼈 및 몸통 주변으로 뻗은 신경 가지망
        for y in range(280, 600, 40):
            draw.line([400, y, 320, y + 25], fill=nerve_color, width=3)
            draw.line([400, y, 480, y + 25], fill=nerve_color, width=3)
            draw.line([320, y + 25, 280, y + 50], fill=nerve_color, width=2)
            draw.line([480, y + 25, 520, y + 50], fill=nerve_color, width=2)
            
        img.save(nervous_torso_path)

    
    # 4.1 순환계 몸통 및 심장 (circulatory_torso.png)
    circulatory_torso_path = "primary_data/circulatory_torso.png"
    if not os.path.exists(circulatory_torso_path):
        img = Image.new("RGBA", (800, 1000), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        art_color = (235, 65, 65, 255) # 동맥 빨강
        vein_color = (65, 115, 225, 255) # 정맥 파랑
        
        # 심장 입체 묘사 (Y좌표 290~370, X좌표 365~445로 상향하고 왼쪽 편향 조정하여 폐 사이 정확히 위치)
        draw.ellipse([365, 290, 445, 370], fill=(225, 50, 50, 255))
        draw.ellipse([380, 302, 430, 352], fill=(255, 90, 90, 255)) # 입체 광택 하이라이트
        # 대동맥 활 (Y=270~300으로 심장 상단 연결)
        draw.arc([390, 272, 430, 302], start=180, end=360, fill=(245, 75, 75, 255), width=10)
        
        # 몸통 혈관 세로 기둥선들
        draw.line([390, 290, 390, 660], fill=art_color, width=10)
        draw.line([410, 290, 410, 660], fill=vein_color, width=10)
        
        # 몸통 주변 모세혈관 가지망 (심장 바로 아래부터 시작)
        for y in range(370, 630, 45):
            draw.line([390, y, 310, y + 20], fill=(235, 65, 65, 180), width=3)
            draw.line([410, y, 490, y + 20], fill=(65, 115, 225, 180), width=3)
            
        img.save(circulatory_torso_path)

    # 5.1 고도화된 입체 소화계 (digestive.png)
    digestive_path = "primary_data/digestive.png"
    if not os.path.exists(digestive_path):
        img = Image.new("RGBA", (800, 1000), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 입체 식도 (Y=200~400)
        draw.rounded_rectangle([390, 200, 410, 400], radius=5, fill=(255, 200, 120, 255))
        draw.rounded_rectangle([393, 200, 407, 400], radius=3, fill=(255, 220, 150, 255)) # 음영
        
        # 입체 간 (Y=390~470 범위로 횡격막 바로 아래 오른쪽에 밀착 배치)
        draw.polygon([(270, 410), (410, 390), (390, 470)], fill=(130, 45, 45, 245)) # 베이스 어두운면
        draw.polygon([(280, 412), (400, 398), (385, 460)], fill=(175, 75, 75, 245)) # 위쪽 밝은면
        
        # 입체 위 (Y=390~470 범위로 횡격막 바로 아래 왼쪽에 배치하고 식도와 연결)
        draw.ellipse([380, 390, 480, 470], fill=(235, 105, 55, 245))
        draw.ellipse([395, 405, 465, 455], fill=(255, 145, 85, 245)) # 부피 하이라이트
        draw.line([395, 390, 395, 415], fill=(235, 105, 55, 245), width=15)
        
        # 입체 대장 (Y=470~620 범위로 간과 위의 아래쪽에 매칭하여 중첩 분리)
        colon_color = (115, 135, 85, 245)
        colon_light = (145, 165, 115, 245)
        # 가로 대장 입체 묘사
        draw.rounded_rectangle([310, 475, 490, 505], radius=10, fill=colon_color)
        draw.rounded_rectangle([320, 480, 480, 498], radius=6, fill=colon_light)
        # 내려가는 대장 좌우
        draw.rounded_rectangle([300, 495, 330, 600], radius=10, fill=colon_color)
        draw.rounded_rectangle([470, 495, 500, 600], radius=10, fill=colon_color)
        draw.rounded_rectangle([385, 590, 415, 625], radius=5, fill=colon_color)
        
        # 소장 (대장 내부 공간인 Y=500~585에 집중 배치)
        small_color = (255, 165, 135, 255)
        small_light = (255, 195, 175, 255)
        for x, y in [(360, 510), (410, 510), (350, 545), (400, 545), (375, 528), (425, 528)]:
            draw.ellipse([x, y, x + 40, y + 35], fill=small_color)
            draw.ellipse([x + 8, y + 8, x + 32, y + 27], fill=small_light) # 하이라이트
            
        img.save(digestive_path)

    # 5.2 고도화된 입체 호흡계 (respiratory.png)
    respiratory_path = "primary_data/respiratory.png"
    if not os.path.exists(respiratory_path):
        img = Image.new("RGBA", (800, 1000), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        trachea_color = (130, 195, 235, 255)
        
        # 기도 뼈 고리 입체화 (위치 상향 보정으로 가슴 상단 매칭)
        draw.rounded_rectangle([388, 180, 412, 270], radius=5, fill=trachea_color)
        for y in range(190, 260, 15):
            draw.line([388, y, 412, y], fill=(255, 255, 255, 200), width=4)
        draw.line([400, 270, 355, 315], fill=trachea_color, width=12)
        draw.line([400, 270, 445, 315], fill=trachea_color, width=12)
        
        # 둥글고 입체적인 폐 (Y=260~400 범위로 횡격막 위, 갈비뼈 내부에 완벽 고정)
        lung_color = (245, 135, 155, 215)
        lung_light = (255, 175, 195, 225)
        # 왼쪽 폐
        draw.ellipse([425, 260, 535, 400], fill=lung_color)
        draw.ellipse([445, 280, 515, 380], fill=lung_light)
        # 오른쪽 폐
        draw.ellipse([265, 260, 375, 400], fill=lung_color)
        draw.ellipse([285, 280, 355, 380], fill=lung_light)
        
        img.save(respiratory_path)

    # 5.3 고도화된 입체 배설계 (excretory.png)
    excretory_path = "primary_data/excretory.png"
    if not os.path.exists(excretory_path):
        img = Image.new("RGBA", (800, 1000), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        kidney = (110, 35, 35, 245)
        kidney_light = (150, 60, 60, 245)
        ureter = (225, 215, 105, 255)
        
        # 신장 (Y=450~510 범위로 위 뒤쪽 척추 양옆 고증 매칭)
        draw.ellipse([440, 450, 480, 510], fill=kidney)
        draw.ellipse([445, 460, 475, 500], fill=kidney_light)
        draw.ellipse([320, 450, 360, 510], fill=kidney)
        draw.ellipse([325, 460, 355, 500], fill=kidney_light)
        
        # 방광 (Y=630~670 범위로 골반강 하단 배치)
        draw.ellipse([375, 630, 425, 670], fill=(235, 195, 55, 245))
        draw.ellipse([385, 635, 415, 662], fill=(255, 225, 95, 245))
        
        # 요관
        draw.line([350, 500, 390, 635], fill=ureter, width=4)
        draw.line([450, 500, 410, 635], fill=ureter, width=4)
        img.save(excretory_path)

    # 안내 가이드라인
    if not os.path.exists(assets["guide"]):
        img = Image.new("RGBA", (400, 600), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        guide_color = (74, 222, 128, 255)
        draw.ellipse([160, 60, 240, 140], outline=guide_color, width=4)
        draw.line([200, 140, 200, 400], fill=guide_color, width=3)
        draw.line([120, 180, 280, 180], fill=guide_color, width=4)
        draw.line([145, 370, 255, 370], fill=guide_color, width=4)
        draw.line([120, 180, 95, 330], fill=guide_color, width=3)
        draw.line([280, 180, 305, 330], fill=guide_color, width=3)
        draw.line([150, 370, 140, 550], fill=guide_color, width=4)
        draw.line([250, 370, 260, 550], fill=guide_color, width=4)
        draw.rounded_rectangle([60, 10, 340, 45], radius=5, outline=guide_color, width=2)
        img.save(assets["guide"])

    # 호흡 및 소화 머리 에셋에 두개골 뼈대 배경 합성하여 얼굴 형태 복구
    try:
        skeleton_head_path = "primary_data/skeleton_head.png"
        if os.path.exists(skeleton_head_path):
            skeleton_img = cv2.imread(skeleton_head_path, cv2.IMREAD_UNCHANGED)
            if skeleton_img is not None:
                for head_name in ["respiratory_head.png", "digestive_head.png"]:
                    head_p = os.path.join("primary_data", head_name)
                    if os.path.exists(head_p):
                        organ_head = cv2.imread(head_p, cv2.IMREAD_UNCHANGED)
                        if organ_head is not None:
                            out_h = skeleton_img.copy()
                            mask_h = organ_head[:, :, 3] > 0
                            out_h[mask_h] = organ_head[mask_h]
                            cv2.imwrite(head_p, out_h)
    except Exception as e:
        pass

# 에셋 초기화 구동
initialize_assets()

# ==========================================
# 2. 이미지 처리 및 과학적 정면 판별 엔진
# ==========================================

import mediapipe as mp
mp_pose = mp.solutions.pose

class DummyLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class DummyLandmarkList:
    def __init__(self, landmarks_dict):
        self.landmark = landmarks_dict

def get_fallback_landmarks(w, h):
    landmarks_dict = {}
    
    # 랜드마크 데이터베이스에 sample_model.png 가 있다면 실제 좌표를 기반으로 비율 환산 생성
    sample_lms = ASSET_LANDMARKS.get("sample_model.png", {})
    if sample_lms:
        for idx_str, pt in sample_lms.items():
            idx = int(idx_str)
            landmarks_dict[idx] = DummyLandmark(pt[0] / 800.0, pt[1] / 1200.0, pt[2] if len(pt) > 2 else 0.0)
        return DummyLandmarkList(landmarks_dict)
        
    # 데이터베이스가 없거나 누락된 경우의 기존 fallback
    # 머리 랜드마크
    landmarks_dict[0] = DummyLandmark(400/800, 155/1200) # 코
    landmarks_dict[7] = DummyLandmark(460/800, 140/1200) # 좌측 귀
    landmarks_dict[8] = DummyLandmark(340/800, 140/1200) # 우측 귀
    landmarks_dict[1] = DummyLandmark(415/800, 135/1200)
    landmarks_dict[2] = DummyLandmark(425/800, 135/1200)
    landmarks_dict[3] = DummyLandmark(435/800, 135/1200)
    landmarks_dict[4] = DummyLandmark(385/800, 135/1200)
    landmarks_dict[5] = DummyLandmark(375/800, 135/1200)
    landmarks_dict[6] = DummyLandmark(365/800, 135/1200)
    landmarks_dict[9] = DummyLandmark(410/800, 175/1200)
    landmarks_dict[10] = DummyLandmark(390/800, 175/1200)
    
    # 어깨
    landmarks_dict[11] = DummyLandmark(550/800, 255/1200) # 좌측 어깨
    landmarks_dict[12] = DummyLandmark(250/800, 255/1200) # 우측 어깨
    
    # 팔 관절
    landmarks_dict[13] = DummyLandmark(610/800, 415/1200) # 좌측 팔꿈치
    landmarks_dict[14] = DummyLandmark(190/800, 415/1200) # 우측 팔꿈치
    landmarks_dict[15] = DummyLandmark(650/800, 560/1200) # 좌측 손목
    landmarks_dict[16] = DummyLandmark(150/800, 560/1200) # 우측 손목
    
    # 골반
    landmarks_dict[23] = DummyLandmark(480/800, 680/1200) # 좌측 골반
    landmarks_dict[24] = DummyLandmark(320/800, 680/1200) # 우측 골반
    
    # 다리 관절
    landmarks_dict[25] = DummyLandmark(480/800, 875/1200) # 좌측 무릎
    landmarks_dict[26] = DummyLandmark(320/800, 875/1200) # 우측 무릎
    landmarks_dict[27] = DummyLandmark(480/800, 1070/1200) # 좌측 발목
    landmarks_dict[28] = DummyLandmark(320/800, 1070/1200) # 우측 발목
    
    return DummyLandmarkList(landmarks_dict)

def inject_virtual_head_landmarks(lms_dict):
    """
    랜드마크 딕셔너리에 가상 정수리(101)와 가상 턱(102)을 계산하여 주입합니다.
    """
    if 7 in lms_dict and 8 in lms_dict:
        p7 = lms_dict[7]
        p8 = lms_dict[8]
        
        # 귀의 중심점
        cx = (p7.x + p8.x) / 2.0
        cy = (p7.y + p8.y) / 2.0
        cz = (p7.z + p8.z) / 2.0
        
        # 가로 방향 거리
        dx = p7.x - p8.x
        dy = p7.y - p8.y
        L = math.sqrt(dx**2 + dy**2)
        
        if L > 0:
            # 수직 위 방향 단위 벡터
            ux_up = dy / L
            uy_up = -dx / L
            
            # 수직 아래 방향 단위 벡터
            ux_down = -dy / L
            uy_down = dx / L
            
            # 101번 정수리는 귀 중심에서 위로 L * 0.53 만큼 이동
            lms_dict[101] = DummyLandmark(cx + ux_up * (L * 0.53), cy + uy_up * (L * 0.53), cz)
            # 102번 턱은 귀 중심에서 아래로 L * 0.42 만큼 이동 (턱선을 목 위쪽으로 복원)
            lms_dict[102] = DummyLandmark(cx + ux_down * (L * 0.42), cy + uy_down * (L * 0.42), cz)
            
    # 입꼬리가 누락된 경우 예외 처리
    if 9 not in lms_dict:
        lms_dict[9] = DummyLandmark(0.0, 0.0)
    if 10 not in lms_dict:
        lms_dict[10] = DummyLandmark(0.0, 0.0)

def extract_body_landmarks(image_rgb):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
    return results.pose_landmarks

def analyze_symmetry_front(landmarks, w, h):
    if landmarks is None or not isinstance(landmarks.landmark, dict) and not hasattr(landmarks, 'landmark'):
        return False, ["랜드마크 정보가 부족하여 자세 분석이 어렵습니다."]
        
    warnings = []
    is_front = True
    
    if isinstance(landmarks.landmark, dict):
        lms = landmarks.landmark
    else:
        lms = {i: landmarks.landmark[i] for i in range(len(landmarks.landmark)) if i < len(landmarks.landmark)}

    if 11 in lms and 12 in lms:
        z_11 = lms[11].z
        z_12 = lms[12].z
        shoulder_diff_z = abs(z_11 - z_12)
        if shoulder_diff_z > 0.13:
            is_front = False
            warnings.append(f"어깨가 회전되어 정면이 아닙니다. (깊이 편차: {shoulder_diff_z:.2f})")
            
    if 23 in lms and 24 in lms:
        z_23 = lms[23].z
        z_24 = lms[24].z
        hip_diff_z = abs(z_23 - z_24)
        if hip_diff_z > 0.13:
            is_front = False
            warnings.append(f"골반이 기울어져 정면이 아닙니다. (깊이 편차: {hip_diff_z:.2f})")
            
    if 0 in lms and 7 in lms and 8 in lms:
        nose_x = lms[0].x * w
        le_x = lms[7].x * w
        re_x = lms[8].x * w
        dist_left = abs(nose_x - le_x)
        dist_right = abs(nose_x - re_x)
        ratio = dist_left / (dist_right + 1e-5)
        if ratio < 0.55 or ratio > 1.8:
            is_front = False
            warnings.append(f"얼굴이 측면을 바라보고 있습니다. (좌우 귀 대칭 비율: {ratio:.2f})")
            
    return is_front, warnings

def generate_face_mask(image_shape, landmarks):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if landmarks is None:
        return mask
        
    pts = []
    if isinstance(landmarks.landmark, dict):
        lms = landmarks.landmark
    else:
        lms = {i: landmarks.landmark[i] for i in range(len(landmarks.landmark)) if i < len(landmarks.landmark)}

    for i in range(11):
        if i in lms:
            pts.append([int(lms[i].x * w), int(lms[i].y * h)])
            
    if 7 in lms and 8 in lms and 11 in lms and 12 in lms:
        le = lms[7]
        re = lms[8]
        ls = lms[11]
        rs = lms[12]
        le_x, le_y = int(le.x * w), int(le.y * h)
        re_x, re_y = int(re.x * w), int(re.y * h)
        ls_x, ls_y = int(ls.x * w), int(ls.y * h)
        rs_x, rs_y = int(rs.x * w), int(rs.y * h)
        pts.append([int(0.65 * le_x + 0.35 * ls_x), int(0.65 * le_y + 0.35 * ls_y)])
        pts.append([int(0.65 * re_x + 0.35 * rs_x), int(0.65 * re_y + 0.35 * rs_y)])
        
    if 0 in lms and 7 in lms and 8 in lms:
        nose = lms[0]
        le = lms[7]
        re = lms[8]
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        ear_cx = int((le.x + re.x) * 0.5 * w)
        ear_cy = int((le.y + re.y) * 0.5 * h)
        pts.append([nose_x - int((ear_cx - nose_x) * 1.3), nose_y - int((ear_cy - nose_y) * 1.3)])
        
    if len(pts) > 0:
        pts_arr = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(pts_arr)
        cv2.fillConvexPoly(mask, hull, 255)
        
    return mask

# ==========================================
# 3. 고도화된 하이브리드 합성 수학 알고리즘
# ==========================================

def transform_body_torso_perspective(organ_path, image_shape, landmarks):
    """
    3개의 기준점(우측어깨, 좌측어깨, 골반중심)을 기반으로 가로세로 비율이 유지되는
    유사 아핀 변환(Similarity Affine Warp)을 가하여 몸통 및 장기 기관들을 인물 체형에 맞춰 정합시킵니다.
    """
    h, w = image_shape[:2]
    organ_img = cv2.imread(organ_path, cv2.IMREAD_UNCHANGED)
    if organ_img is None:
        return np.zeros((h, w, 4), dtype=np.uint8)
        
    # BGR/BGRA 에서 RGB/RGBA 채널 변환 적용
    if organ_img.shape[2] == 4:
        organ_img = cv2.cvtColor(organ_img, cv2.COLOR_BGRA2RGBA)
    elif organ_img.shape[2] == 3:
        organ_img = cv2.cvtColor(organ_img, cv2.COLOR_BGR2RGB)
        
    filename = os.path.basename(organ_path)
    lms_src = ASSET_LANDMARKS.get(filename, {})
    
    # 에셋 소스 3점 정의 (우측어깨 12번, 좌측어깨 11번, 골반 중심 23/24번 평균)
    # skeleton_torso.png 기준 랜드마크: 11번 (356.0, 95.0), 12번 (52.0, 95.0), 23번 (310.0, 690.0), 24번 (98.0, 690.0)
    # 골반 중심: X: 204.0, Y: 690.0
    if lms_src and "11" in lms_src and "12" in lms_src and "23" in lms_src and "24" in lms_src:
        p11_s = np.array(lms_src["11"][:2])
        p12_s = np.array(lms_src["12"][:2])
        p23_s = np.array(lms_src["23"][:2])
        p24_s = np.array(lms_src["24"][:2])
        
        src_pts = np.float32([
            p12_s, # 우측 어깨
            p11_s, # 좌측 어깨
            (p23_s + p24_s) / 2.0 # 골반 중심
        ])
    else:
        src_pts = np.float32([
            [52.0, 95.0],   # 우측 어깨
            [356.0, 95.0],  # 좌측 어깨
            [204.0, 690.0]  # 골반 중심
        ])
        
    if isinstance(landmarks.landmark, dict):
        lms = landmarks.landmark
    else:
        lms = {i: landmarks.landmark[i] for i in range(len(landmarks.landmark)) if i < len(landmarks.landmark)}
        
    ls = lms[11]
    rs = lms[12]
    
    # 타깃 3점 설정
    ls_pt = np.array([ls.x * w, ls.y * h])
    rs_pt = np.array([rs.x * w, rs.y * h])
    
    # 골반 점 검증 및 가상 골반 중심 보간
    has_hip = (23 in lms and 24 in lms and hasattr(lms[23], 'visibility') and lms[23].visibility > 0.55 and lms[24].visibility > 0.55)
    if has_hip:
        lh_pt = np.array([lms[23].x * w, lms[23].y * h])
        rh_pt = np.array([lms[24].x * w, lms[24].y * h])
        hip_pt = (lh_pt + rh_pt) / 2.0
    else:
        # 어깨 중심과 방향 벡터를 활용한 가상 골반 보간
        cx = (ls_pt[0] + rs_pt[0]) / 2.0
        cy = (ls_pt[1] + rs_pt[1]) / 2.0
        sh_left_x = min(rs_pt[0], ls_pt[0])
        sh_right_x = max(rs_pt[0], ls_pt[0])
        sh_left_y = rs_pt[1] if rs_pt[0] < ls_pt[0] else ls_pt[1]
        sh_right_y = ls_pt[1] if rs_pt[0] < ls_pt[0] else rs_pt[1]
        
        v_dx = sh_right_x - sh_left_x
        v_dy = sh_right_y - sh_left_y
        v_actual_w = math.sqrt(v_dx*v_dx + v_dy*v_dy)
        
        ux_down = -v_dy / (v_actual_w + 1e-5)
        uy_down = v_dx / (v_actual_w + 1e-5)
        virtual_h = v_actual_w * 1.38
        hip_pt = np.array([cx + ux_down * virtual_h, cy + uy_down * virtual_h])
        
    dst_pts = np.float32([
        rs_pt,
        ls_pt,
        hip_pt
    ])
    
    # 유사 변환 행렬 M 산출 (가로세로 비례가 동일하게 유지됨)
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        # 실패 시 폴백으로 일반 아핀 변환 유도
        M = cv2.getAffineTransform(src_pts, dst_pts)
        
    warped = cv2.warpAffine(
        organ_img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return warped

def transform_limb_segment(segment_path, image_shape, start_pt, end_pt, shoulder_width):
    """
    팔다리 마디 전용 에셋(150x400)을 실제 관절 랜드마크(시작점 및 끝점) 벡터에 일치하도록
    기하학적 복합 변환 행렬(스케일 + 회전 + 평행이동)을 계산해 합성합니다.
    """
    h, w = image_shape[:2]
    seg_img = cv2.imread(segment_path, cv2.IMREAD_UNCHANGED)
    if seg_img is None:
        return np.zeros((h, w, 4), dtype=np.uint8)
        
    xs, ys = start_pt
    xe, ye = end_pt
    
    dx = xe - xs
    dy = ye - ys
    len_real = math.sqrt(dx*dx + dy*dy)
    angle = math.atan2(dy, dx)
    
    len_src = 400.0
    s_y = len_real / len_src
    
    s_x = (shoulder_width / 300.0) * 0.95
    if s_x < 0.2: s_x = 0.2
    if s_x > 2.0: s_x = 2.0
    
    rot_angle = angle - math.pi / 2
    
    T1 = np.float32([
        [1, 0, -75],
        [0, 1, -20],
        [0, 0, 1]
    ])
    
    S = np.float32([
        [s_x, 0, 0],
        [0, s_y, 0],
        [0, 0, 1]
    ])
    
    cos_a = math.cos(rot_angle)
    sin_a = math.sin(rot_angle)
    R = np.float32([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    T2 = np.float32([
        [1, 0, xs],
        [0, 1, ys],
        [0, 0, 1]
    ])
    
    matrix_3x3 = T2 @ R @ S @ T1
    matrix = matrix_3x3[:2, :]
    
    warped = cv2.warpAffine(
        seg_img, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return warped

def get_hybrid_overlay_layer(image_shape, landmarks, organ_type, head_scale=1.0):
    """
    골격계, 근육계, 신경계, 순환계, 호흡계, 소화계의 머리와 몸통 오버레이를
    3점 유사 아핀 변환으로 찌그러짐 없이 입체적이고 정확하게 정합합니다.
    """
    h, w = image_shape[:2]
    
    if landmarks is None:
        landmarks = get_fallback_landmarks(w, h)
        
    if isinstance(landmarks.landmark, dict):
        lms = landmarks.landmark
    else:
        lms = {i: landmarks.landmark[i] for i in range(len(landmarks.landmark)) if i < len(landmarks.landmark)}
        
    if organ_type == "skeleton":
        torso_path = "primary_data/skeleton_torso.png"
    elif organ_type in ["respiratory", "digestive", "excretory"]:
        torso_path = f"primary_data/{organ_type}.png"
    else:
        torso_path = f"primary_data/{organ_type}_torso.png"
        
    # 몸통 레이어 3점 유사 아핀 변환 적용
    torso_layer = transform_body_torso_perspective(torso_path, image_shape, landmarks)
    
    if organ_type in ["skeleton", "nervous", "muscles", "circulatory", "respiratory", "digestive"]:
        head_file = f"{organ_type}_head.png"
        lookup_file = "skeleton_head.png" if head_file in ["respiratory_head.png", "digestive_head.png"] else head_file
        lms_src = ASSET_LANDMARKS.get(lookup_file, {})
        
        # 머리는 우안외측(6번), 좌안외측(3번), 입꼬리 중심(9, 10번 평균) 3점 정밀 유사 변환 매핑
        # 에셋 소스 3점 추출
        if lms_src and "3" in lms_src and "6" in lms_src and "9" in lms_src and "10" in lms_src:
            p3_s = np.array(lms_src["3"][:2])
            p6_s = np.array(lms_src["6"][:2])
            p9_s = np.array(lms_src["9"][:2])
            p10_s = np.array(lms_src["10"][:2])
            
            src_head_pts = np.float32([
                p6_s, # 우안외측 (이미지 상 왼쪽 눈)
                p3_s, # 좌안외측 (이미지 상 오른쪽 눈)
                (p9_s + p10_s) / 2.0 # 입꼬리 중심
            ])
        else:
            # 예외 폴백 (262x300 에셋 해상도 기준 복원 좌표)
            src_head_pts = np.float32([
                [79.38, 171.47],  # 우안외측
                [181.21, 171.81], # 좌안외측
                [135.05, 263.41]  # 입꼬리 중심
            ])
            
        # 타깃 3점 추출 (사용자 자세 인식 랜드마크 기반)
        if 3 in lms and 6 in lms and 9 in lms and 10 in lms:
            p3_t = np.array([lms[3].x * w, lms[3].y * h])
            p6_t = np.array([lms[6].x * w, lms[6].y * h])
            p9_t = np.array([lms[9].x * w, lms[9].y * h])
            p10_t = np.array([lms[10].x * w, lms[10].y * h])
            
            dst_head_pts = np.float32([
                p6_t, # 우안외측
                p3_t, # 좌안외측
                (p9_t + p10_t) / 2.0 # 입꼬리 중심
            ])
            
            # 유사 변환 행렬 M 계산
            M_head, inliers = cv2.estimateAffinePartial2D(src_head_pts, dst_head_pts)
            if M_head is None:
                M_head = cv2.getAffineTransform(src_head_pts, dst_head_pts)
                
            if head_scale != 1.0 and M_head is not None:
                tcx = np.mean(dst_head_pts[:, 0])
                tcy = np.mean(dst_head_pts[:, 1])
                M_head[0, 0] *= head_scale
                M_head[0, 1] *= head_scale
                M_head[1, 0] *= head_scale
                M_head[1, 1] *= head_scale
                M_head[0, 2] = M_head[0, 2] * head_scale + tcx * (1.0 - head_scale)
                M_head[1, 2] = M_head[1, 2] * head_scale + tcy * (1.0 - head_scale)
                
            head_path = f"primary_data/{head_file}"
            head_img = cv2.imread(head_path, cv2.IMREAD_UNCHANGED)
            if head_img is not None:
                head_img = cv2.cvtColor(head_img, cv2.COLOR_BGRA2RGBA)
                
                # 호흡 및 소화계 머리 장기일 경우 메모리 상에서 두개골 뼈대 배경 동적 합성
                if head_file in ["respiratory_head.png", "digestive_head.png"]:
                    sk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "primary_data", "skeleton_head.png"))
                    if os.path.exists(sk_path):
                        sk_img = cv2.imread(sk_path, cv2.IMREAD_UNCHANGED)
                        if sk_img is not None:
                            sk_img = cv2.cvtColor(sk_img, cv2.COLOR_BGRA2RGBA)
                            if sk_img.shape[:2] != head_img.shape[:2]:
                                head_img = cv2.resize(head_img, (sk_img.shape[1], sk_img.shape[0]), interpolation=cv2.INTER_AREA)
                            out_h = sk_img.copy()
                            mask_h = head_img[:, :, 3] > 0
                            out_h[mask_h] = head_img[mask_h]
                            head_img = out_h
                head_layer = cv2.warpAffine(
                    head_img, M_head, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)
                )
                
                # 몸통 레이어 위에 머리 레이어 오버레이 합성 (RGBA 알파 채널 오버랩)
                head_rgb = head_layer[:, :, :3]
                head_alpha = np.expand_dims(head_layer[:, :, 3] / 255.0, axis=2)
                torso_layer[:, :, :3] = (head_rgb * head_alpha + torso_layer[:, :, :3] * (1.0 - head_alpha)).astype(np.uint8)
                torso_layer[:, :, 3] = np.maximum(torso_layer[:, :, 3], head_layer[:, :, 3])
                
    return torso_layer
            
    # 몸통 레이어와 모든 팔다리 마디 레이어들을 RGBA 알파 블렌딩으로 통합
    combined = torso_layer.copy()
    for limb in limb_layers:
        limb_rgb = limb[:, :, :3]
        limb_alpha = np.expand_dims(limb[:, :, 3] / 255.0, axis=2)
        combined[:, :, :3] = (limb_rgb * limb_alpha + combined[:, :, :3] * (1.0 - limb_alpha)).astype(np.uint8)
        combined[:, :, 3] = np.maximum(combined[:, :, 3], limb[:, :, 3])
        
    return combined

def build_final_overlay(background_rgb, organ_layers, selected_keys, selected_weights):
    """
    다중 장기 레이어를 순서대로 결합하되, 
    각 레이어별 가중치(슬라이더 값)를 동적으로 반영하여 블렌딩합니다.
    """
    output = background_rgb.copy()
    
    for layer, key, weight in zip(organ_layers, selected_keys, selected_weights):
        layer_rgb = layer[:, :, :3]
        layer_alpha = layer[:, :, 3] / 255.0
        
        # 슬라이더로 조절된 동적 투명도 가중치 반영
        layer_alpha_weighted = layer_alpha * weight
            
        alpha_factor = np.expand_dims(layer_alpha_weighted, axis=2)
        output = (layer_rgb * alpha_factor + output * (1.0 - alpha_factor)).astype(np.uint8)
        
    return output

# ==========================================
# 4. Streamlit 인터페이스 및 교육 뷰
# ==========================================

import base64

def get_base64_image_uri(filename):
    path = f"primary_data/{filename}"
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as image_file:
        return "data:image/png;base64," + base64.b64encode(image_file.read()).decode('utf-8')

def get_base64_image_from_numpy(img_rgb):
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if is_success:
        return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
    return ""

# 페이지 설정 (단일 칼럼 반응형 최적화를 위해 layout="wide"를 유지하되 내부 분할 조율)
st.set_page_config(
    page_title="신기한 우리 몸 탐험: 5학년 과학교육 프로그램",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 7대 신체 기관 정보 정의 (키, 간략화된 한글명)
organs = [
    ("skeleton", "골격"),
    ("muscles", "근육"),
    ("nervous", "신경"),
    ("circulatory", "순환"),
    ("respiratory", "호흡"),
    ("digestive", "소화"),
    ("excretory", "배설"),
]

# 세션 상태 변수 초기화
if "active_organ" not in st.session_state:
    st.session_state["active_organ"] = None
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "시작 화면"

active_organ = st.session_state["active_organ"]
current_mode = st.session_state["app_mode"]

# Premium UI CSS 주입 (모던 웹 프로그램 대시보드 및 글래스모피즘 스타일)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Nanum+Gothic:wght@400;700;800&display=swap');
    
    * {
        font-family: 'Nanum Gothic', 'Outfit', sans-serif;
    }
    
    /* 메인 컨테이너 패딩 조절 */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100vw !important;
    }
    
    /* 시작 화면 인트로 디자인 */
    .intro-header {
        text-align: center;
        margin-top: 4rem;
        margin-bottom: 3rem;
    }
    
    .intro-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0284c7 0%, #3b82f6 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
    }
    
    .intro-subtitle {
        font-size: 1.4rem;
        color: #64748b;
        font-weight: 400;
    }
    
    /* 인트로 카드형 통합 버튼 커스텀 스타일 */
    div[data-testid="column"] button:has(div:contains("기본 모델")),
    div[data-testid="column"] button:has(div:contains("사진 업로드")),
    div[data-testid="column"] button:has(div:contains("실시간 카메라")) {
        height: 280px !important;
        background: rgba(255, 255, 255, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 24px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: #1e293b !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        white-space: pre-line !important;
        cursor: pointer !important;
        gap: 15px !important;
    }
    
    div[data-testid="column"] button:has(div:contains("기본 모델")):hover,
    div[data-testid="column"] button:has(div:contains("사진 업로드")):hover,
    div[data-testid="column"] button:has(div:contains("실시간 카메라")):hover {
        transform: translateY(-8px) !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1) !important;
        background: rgba(255, 255, 255, 0.65) !important;
        border-color: #3b82f6 !important;
    }
    
    /* 카드 설명 캡션 스타일 */
    .card-caption {
        text-align: center;
        font-size: 0.95rem;
        color: #64748b;
        line-height: 1.5;
        margin-top: 0.8rem;
        padding: 0 1rem;
    }
    
    /* 네비게이션 헤더바 디자인 */
    .nav-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.03);
        padding: 0.8rem 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        margin-bottom: 1.2rem;
    }
    
    .nav-title {
        font-size: 1.35rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0284c7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 사이드 컨트롤 보드 디자인 */
    .sidebar-board {
        background: rgba(15, 23, 42, 0.02);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 20px;
        padding: 1.2rem;
    }
    
    /* 활성화 버튼 강조 스타일 */
    div[data-testid="stButton"] button:has(div:contains("🟢")),
    div[data-testid="stButton"] button:contains("🟢") {
        background: linear-gradient(135deg, #0284c7 0%, #3b82f6 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        transform: translateY(-1px);
    }
    
    /* 비활성화 버튼 흐림 스타일 */
    div[data-testid="stButton"] button:has(div:contains("⚪")),
    div[data-testid="stButton"] button:contains("⚪") {
        background-color: #ffffff !important;
        color: #64748b !important;
        opacity: 0.65 !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    div[data-testid="stButton"] button:hover {
        opacity: 1.0 !important;
        border-color: #3b82f6 !important;
    }
    
    /* 반응형 비디오 및 캔버스 크기 제어 */
    #canvas-container {
        position: relative;
        width: 100%;
        max-width: 640px;
        margin: 0 auto;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        background-color: #000000;
    }
    
    #ar-canvas {
        width: 100%;
        height: auto;
        display: block;
        transform: scaleX(-1);
    }
    
    /* Deploy 단추, 햄버거 메뉴, 헤더 푸터 등 프레임 완전 소거 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stHeader"] {display: none;}
    div[data-testid="stDecoration"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 5. 분기형 뷰 렌더링 구조화
# ==========================================

if current_mode == "시작 화면":
    # 시작 화면 렌더링
    st.markdown("""
    <div class="intro-header">
        <div class="intro-title">🧬 신기한 우리 몸 속 탐험</div>
        <div class="intro-subtitle">초등학교 5학년 과학: 우리 몸의 구조와 기능의 유기적인 연동</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3대 모드 수평 카드 레이아웃
    card_cols = st.columns(3, gap="large")
    
    modes_data = [
        ("🙋\n\n기본 모델", "초기 설정된 가상 3D 캐릭터 모델 위에 신체 장기들을 정밀 결합하여 탐구합니다.", "btn_card_base"),
        ("📸\n\n사진 업로드", "본인의 전신 정면 사진을 올려 내 실제 신체 비율 위에 장기를 합성해봅니다.", "btn_card_photo"),
        ("📹\n\n실시간 카메라", "카메라를 켜고 몸을 움직이면 관절 인식 기술을 통해 장기가 나를 따라 실시간 정합됩니다.", "btn_card_cam")
    ]
    
    for idx, (display_title, desc, btn_key) in enumerate(modes_data):
        with card_cols[idx]:
            # 카드 버튼 자체로 시작 화면 3대 모드를 즉시 클릭 실행하도록 통합
            if st.button(display_title, key=btn_key, use_container_width=True):
                # 텍스트에 매칭하여 세션 상태 변경
                target_mode = "기본 모델" if "기본 모델" in display_title else ("사진 업로드" if "사진 업로드" in display_title else "실시간 카메라")
                st.session_state["app_mode"] = target_mode
                st.rerun()
            # 카드 하단에 설명글 매핑
            st.markdown(f'<div class="card-caption">{desc}</div>', unsafe_allow_html=True)

else:
    # 홈으로 복귀하는 별도 단추 단독 구성
    header_col1, header_col2 = st.columns([7, 1])
    with header_col2:
        if st.button("↩️ 처음으로", key="btn_go_home", use_container_width=True):
            st.session_state["app_mode"] = "시작 화면"
            st.session_state["active_organ"] = None  # 리셋
            st.rerun()
            
    st.markdown("<div style='margin-top: -1.5rem;'></div>", unsafe_allow_html=True)
    
    # 좌우 2분할 레이아웃
    col_left, col_right = st.columns([1, 3], gap="medium")
    
    # 1. 좌측 영역: 사이드 플로팅 기관 선택 보드
    with col_left:
        st.markdown("<div class='sidebar-board'>", unsafe_allow_html=True)
        st.markdown("#### 🫁 기관")
        
        for key, display_name in organs:
            is_active = (active_organ == key)
            icon = "🟢" if is_active else "⚪"
            btn_label = f"{icon} {display_name}"
            
            if st.button(btn_label, key=f"btn_{key}", use_container_width=True):
                if is_active:
                    st.session_state["active_organ"] = None
                else:
                    st.session_state["active_organ"] = key
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
    # 2. 우측 영역: 모니터 뷰포트
    with col_right:
        if current_mode == "기본 모델":
            sample_model_path = "primary_data/sample_model.png"
            if os.path.exists(sample_model_path):
                img_bgr = cv2.imread(sample_model_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
                landmarks = get_fallback_landmarks(w, h)
                
                if active_organ:
                    if active_organ in ["skeleton", "muscles", "circulatory", "nervous", "respiratory", "digestive"]:
                        warped = get_hybrid_overlay_layer(img_rgb.shape, landmarks, active_organ)
                    else:
                        path = f"primary_data/{active_organ}.png"
                        warped = transform_body_torso_perspective(path, img_rgb.shape, landmarks)
                        
                    final_img = build_final_overlay(img_rgb, [warped], [active_organ], [0.85])
                else:
                    final_img = img_rgb
                
                base64_final = get_base64_image_from_numpy(final_img)
                
                # 인터랙티브 줌 및 팬 HTML5 Canvas 컴포넌트 이식
                zoom_html = f"""
                <div id="canvas-container" style="width: 100%; height: 550px; overflow: hidden; position: relative; background: #0f172a; border-radius: 24px; box-shadow: 0 15px 35px rgba(0,0,0,0.25);">
                    <canvas id="zoom-canvas" style="position: absolute; left: 0; top: 0; cursor: move;"></canvas>
                </div>
                <script>
                    const canvas = document.getElementById('zoom-canvas');
                    const ctx = canvas.getContext('2d');
                    const container = document.getElementById('canvas-container');
                    
                    canvas.width = container.clientWidth;
                    canvas.height = container.clientHeight;
                    
                    const img = new Image();
                    img.src = "{base64_final}";
                    
                    let scale = 1.0;
                    let originX = 0;
                    let originY = 0;
                    let isDragging = false;
                    let startX = 0;
                    let startY = 0;
                    
                    img.onload = function() {{
                        const scaleX = canvas.width / img.width;
                        const scaleY = canvas.height / img.height;
                        scale = Math.min(scaleX, scaleY) * 0.95;
                        
                        originX = (canvas.width - img.width * scale) / 2;
                        originY = (canvas.height - img.height * scale) / 2;
                        
                        draw();
                    }};
                    
                    function draw() {{
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.save();
                        ctx.translate(originX, originY);
                        ctx.scale(scale, scale);
                        ctx.drawImage(img, 0, 0);
                        ctx.restore();
                    }}
                    
                    container.addEventListener('wheel', function(e) {{
                        e.preventDefault();
                        const zoomFactor = 1.1;
                        const mouseX = e.clientX - container.getBoundingClientRect().left;
                        const mouseY = e.clientY - container.getBoundingClientRect().top;
                        
                        const imgX = (mouseX - originX) / scale;
                        const imgY = (mouseY - originY) / scale;
                        
                        if (e.deltaY < 0) {{
                            scale *= zoomFactor;
                        }} else {{
                            scale /= zoomFactor;
                        }}
                        scale = Math.max(0.1, Math.min(scale, 10.0));
                        originX = mouseX - imgX * scale;
                        originY = mouseY - imgY * scale;
                        
                        draw();
                    }}, {{passive: false}});
                    
                    canvas.addEventListener('mousedown', function(e) {{
                        isDragging = true;
                        startX = e.clientX - originX;
                        startY = e.clientY - originY;
                    }});
                    
                    window.addEventListener('mouseup', function() {{
                        isDragging = false;
                    }});
                    
                    canvas.addEventListener('mousemove', function(e) {{
                        if (isDragging) {{
                            originX = e.clientX - startX;
                            originY = e.clientY - startY;
                            draw();
                        }}
                    }});
                    
                    // 터치 확대/축소 및 이동 지원
                    let touchStartDist = 0;
                    let isPinching = false;
                    
                    canvas.addEventListener('touchstart', function(e) {{
                        if (e.touches.length === 1) {{
                            isDragging = true;
                            startX = e.touches[0].clientX - originX;
                            startY = e.touches[0].clientY - originY;
                        }} else if (e.touches.length === 2) {{
                            isDragging = false;
                            isPinching = true;
                            const dx = e.touches[0].clientX - e.touches[1].clientX;
                            const dy = e.touches[0].clientY - e.touches[1].clientY;
                            touchStartDist = Math.sqrt(dx*dx + dy*dy);
                        }}
                    }});
                    
                    canvas.addEventListener('touchend', function() {{
                        isDragging = false;
                        isPinching = false;
                    }});
                    
                    canvas.addEventListener('touchmove', function(e) {{
                        if (isDragging && e.touches.length === 1) {{
                            originX = e.touches[0].clientX - startX;
                            originY = e.touches[0].clientY - startY;
                            draw();
                        }} else if (isPinching && e.touches.length === 2) {{
                            const dx = e.touches[0].clientX - e.touches[1].clientX;
                            const dy = e.touches[0].clientY - e.touches[1].clientY;
                            const dist = Math.sqrt(dx*dx + dy*dy);
                            const ratio = dist / touchStartDist;
                            
                            const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2 - container.getBoundingClientRect().left;
                            const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2 - container.getBoundingClientRect().top;
                            
                            const imgX = (midX - originX) / scale;
                            const imgY = (midY - originY) / scale;
                            
                            scale *= ratio;
                            scale = Math.max(0.1, Math.min(scale, 10.0));
                            
                            originX = midX - imgX * scale;
                            originY = midY - imgY * scale;
                            
                            touchStartDist = dist;
                            draw();
                        }}
                    }});
                </script>
                """
                st.components.v1.html(zoom_html, height=570)
            else:
                st.error("기본 모델 이미지가 존재하지 않습니다.")
                
        elif current_mode == "사진 업로드":
            uploaded_file = st.file_uploader(
                "자신의 정면 전신 사진을 업로드해 주세요",
                type=["jpg", "jpeg", "png"],
                key="uploader_viewport"
            )
            
            # 1. 신규 파일이 업로드되었거나 파일 상태가 변경된 경우 세션 캐시 갱신
            if uploaded_file is not None:
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                if (
                    "cached_file_id" not in st.session_state or
                    st.session_state.cached_file_id != file_id or
                    "cached_img_rgb" not in st.session_state or
                    "cached_landmarks" not in st.session_state
                ):
                    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        h, w = img_rgb.shape[:2]
                        
                        landmarks = extract_body_landmarks(img_rgb)
                        is_fallback = False
                        if landmarks is None:
                            landmarks = get_fallback_landmarks(w, h)
                            is_fallback = True
                        
                        st.session_state.cached_file_id = file_id
                        st.session_state.cached_img_rgb = img_rgb
                        st.session_state.cached_landmarks = landmarks
                        st.session_state.cached_is_fallback = is_fallback
            else:
                # 사용자가 업로더에서 업로드된 파일을 명시적으로 제거(X 버튼 클릭)했을 때만 캐시를 폐기
                if "uploader_viewport" in st.session_state and st.session_state.uploader_viewport is None:
                    if "cached_file_id" in st.session_state:
                        del st.session_state.cached_file_id
                    if "cached_img_rgb" in st.session_state:
                        del st.session_state.cached_img_rgb
                    if "cached_landmarks" in st.session_state:
                        del st.session_state.cached_landmarks
                    if "cached_is_fallback" in st.session_state:
                        del st.session_state.cached_is_fallback
            
            # 2. 업로드 파일의 세션 캐시 데이터가 유효하면 렌더링 구동
            if "cached_img_rgb" in st.session_state and "cached_landmarks" in st.session_state:
                img_rgb = st.session_state.cached_img_rgb
                landmarks = st.session_state.cached_landmarks
                is_fallback = st.session_state.get("cached_is_fallback", False)
                h, w = img_rgb.shape[:2]
                
                if is_fallback:
                    st.warning("⚠️ 사진에서 자세를 인식하지 못해 대략적인 기본 위치에 합성합니다. 정면에서 전신이 잘 나오도록 다시 찍은 사진을 추천합니다.")
                else:
                    is_front, warnings_front = analyze_symmetry_front(landmarks, w, h)
                    if not is_front:
                        for msg in warnings_front:
                            st.warning(f"⚠️ {msg} (정밀한 장기 합성을 위해 몸을 정면으로 바르게 선 사진을 권장합니다.)")
                    
                if active_organ:
                    # 모든 기관에 대해 3점 유사 아핀 변환 오버레이 파이프라인으로 통합
                    warped = get_hybrid_overlay_layer(img_rgb.shape, landmarks, active_organ, head_scale=0.9)
                    final_img = build_final_overlay(img_rgb, [warped], [active_organ], [0.85])
                else:
                    final_img = img_rgb
                    
                base64_final = get_base64_image_from_numpy(final_img)
                
                # 인터랙티브 줌 및 팬 HTML5 Canvas 컴포넌트 이식 (사진 업로드)
                zoom_html = f"""
                <div id="canvas-container" style="width: 100%; height: 550px; overflow: hidden; position: relative; background: #0f172a; border-radius: 24px; box-shadow: 0 15px 35px rgba(0,0,0,0.25);">
                    <canvas id="zoom-canvas" style="position: absolute; left: 0; top: 0; cursor: move;"></canvas>
                </div>
                <script>
                    const canvas = document.getElementById('zoom-canvas');
                    const ctx = canvas.getContext('2d');
                    const container = document.getElementById('canvas-container');
                    
                    canvas.width = container.clientWidth;
                    canvas.height = container.clientHeight;
                    
                    const img = new Image();
                    img.src = "{base64_final}";
                    
                    let scale = 1.0;
                    let originX = 0;
                    let originY = 0;
                    let isDragging = false;
                    let startX = 0;
                    let startY = 0;
                    
                    img.onload = function() {{
                        const scaleX = canvas.width / img.width;
                        const scaleY = canvas.height / img.height;
                        scale = Math.min(scaleX, scaleY) * 0.95;
                        
                        originX = (canvas.width - img.width * scale) / 2;
                        originY = (canvas.height - img.height * scale) / 2;
                        
                        draw();
                    }};
                    
                    function draw() {{
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.save();
                        ctx.translate(originX, originY);
                        ctx.scale(scale, scale);
                        ctx.drawImage(img, 0, 0);
                        ctx.restore();
                    }}
                    
                    container.addEventListener('wheel', function(e) {{
                        e.preventDefault();
                        const zoomFactor = 1.1;
                        const mouseX = e.clientX - container.getBoundingClientRect().left;
                        const mouseY = e.clientY - container.getBoundingClientRect().top;
                        
                        const imgX = (mouseX - originX) / scale;
                        const imgY = (mouseY - originY) / scale;
                        
                        if (e.deltaY < 0) {{
                            scale *= zoomFactor;
                        }} else {{
                            scale /= zoomFactor;
                        }}
                        scale = Math.max(0.1, Math.min(scale, 10.0));
                        originX = mouseX - imgX * scale;
                        originY = mouseY - imgY * scale;
                        
                        draw();
                    }}, {{passive: false}});
                    
                    canvas.addEventListener('mousedown', function(e) {{
                        isDragging = true;
                        startX = e.clientX - originX;
                        startY = e.clientY - originY;
                    }});
                    
                    window.addEventListener('mouseup', function() {{
                        isDragging = false;
                    }});
                    
                    canvas.addEventListener('mousemove', function(e) {{
                        if (isDragging) {{
                            originX = e.clientX - startX;
                            originY = e.clientY - startY;
                            draw();
                        }}
                    }});
                    
                    let touchStartDist = 0;
                    let isPinching = false;
                    
                    canvas.addEventListener('touchstart', function(e) {{
                        if (e.touches.length === 1) {{
                            isDragging = true;
                            startX = e.touches[0].clientX - originX;
                            startY = e.touches[0].clientY - originY;
                        }} else if (e.touches.length === 2) {{
                            isDragging = false;
                            isPinching = true;
                            const dx = e.touches[0].clientX - e.touches[1].clientX;
                            const dy = e.touches[0].clientY - e.touches[1].clientY;
                            touchStartDist = Math.sqrt(dx*dx + dy*dy);
                        }}
                    }});
                    
                    canvas.addEventListener('touchend', function() {{
                        isDragging = false;
                        isPinching = false;
                    }});
                    
                    canvas.addEventListener('touchmove', function(e) {{
                        if (isDragging && e.touches.length === 1) {{
                            originX = e.touches[0].clientX - startX;
                            originY = e.touches[0].clientY - startY;
                            draw();
                        }} else if (isPinching && e.touches.length === 2) {{
                            const dx = e.touches[0].clientX - e.touches[1].clientX;
                            const dy = e.touches[0].clientY - e.touches[1].clientY;
                            const dist = Math.sqrt(dx*dx + dy*dy);
                            const ratio = dist / touchStartDist;
                            
                            const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2 - container.getBoundingClientRect().left;
                            const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2 - container.getBoundingClientRect().top;
                            
                            const imgX = (midX - originX) / scale;
                            const imgY = (midY - originY) / scale;
                            
                            scale *= ratio;
                            scale = Math.max(0.1, Math.min(scale, 10.0));
                            
                            originX = midX - imgX * scale;
                            originY = midY - imgY * scale;
                            
                            touchStartDist = dist;
                            draw();
                        }}
                    }});
                </script>
                """
                st.components.v1.html(zoom_html, height=570)
            else:
                st.info("💡 위의 파일 올리기 단추를 사용해 본인의 전신 사진을 올려보세요!")
                
        elif current_mode == "실시간 카메라":
            if active_organ:
                # 선택된 활성 에셋 이미지를 base64 로딩하여 웹캠 JS 로 바로 주입
                torso_file = "skeleton_torso.png" if active_organ == "skeleton" else (f"{active_organ}.png" if active_organ in ["digestive", "respiratory", "excretory"] else f"{active_organ}_torso.png")
                head_file = f"{active_organ}_head.png" if active_organ in ["skeleton", "muscles", "nervous", "circulatory", "respiratory", "digestive"] else None
                
                base64_torso = get_base64_image_uri(torso_file)
                base64_head = ""
                if head_file:
                    if head_file in ["respiratory_head.png", "digestive_head.png"]:
                        sk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "primary_data", "skeleton_head.png"))
                        organ_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "primary_data", head_file))
                        if os.path.exists(sk_path) and os.path.exists(organ_path):
                            sk_img = cv2.imread(sk_path, cv2.IMREAD_UNCHANGED)
                            organ_img = cv2.imread(organ_path, cv2.IMREAD_UNCHANGED)
                            if sk_img is not None and organ_img is not None:
                                if sk_img.shape[:2] != organ_img.shape[:2]:
                                    organ_img = cv2.resize(organ_img, (sk_img.shape[1], sk_img.shape[0]), interpolation=cv2.INTER_AREA)
                                out_h = sk_img.copy()
                                mask_h = organ_img[:, :, 3] > 0
                                out_h[mask_h] = organ_img[mask_h]
                                is_success, buffer = cv2.imencode(".png", out_h)
                                if is_success:
                                    base64_head = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
                                else:
                                    base64_head = get_base64_image_uri(head_file)
                            else:
                                base64_head = get_base64_image_uri(head_file)
                        else:
                            base64_head = get_base64_image_uri(head_file)
                    else:
                        base64_head = get_base64_image_uri(head_file)
                
                # 머리 에셋 정밀 캘리브레이션 랜드마크 추출
                s3_hx, s3_hy = 181.21, 171.81
                s6_hx, s6_hy = 79.38, 171.47
                s_mouth_hx, s_mouth_hy = 135.05, 263.41
                
                if head_file:
                    hlms = ASSET_LANDMARKS.get(head_file, {})
                    if hlms and "3" in hlms and "6" in hlms and "9" in hlms and "10" in hlms:
                        s3_hx, s3_hy = hlms["3"][0], hlms["3"][1]
                        s6_hx, s6_hy = hlms["6"][0], hlms["6"][1]
                        s_mouth_hx = (hlms["9"][0] + hlms["10"][0]) / 2.0
                        s_mouth_hy = (hlms["9"][1] + hlms["10"][1]) / 2.0
                
                # 해당 에셋의 정밀 캘리브레이션 랜드마크 추출 (자바스크립트 3점 동적 매핑)
                lms = ASSET_LANDMARKS.get(torso_file, {})
                if lms and "11" in lms and "12" in lms and "23" in lms and "24" in lms:
                    s11 = lms["11"]
                    s12 = lms["12"]
                    s23 = lms["23"]
                    s24 = lms["24"]
                    s12_x, s12_y = s12[0], s12[1]
                    s11_x, s11_y = s11[0], s11[1]
                    s_hip_x = (s23[0] + s24[0]) / 2.0
                    s_hip_y = (s23[1] + s24[1]) / 2.0
                else:
                    s12_x, s12_y = 52.0, 95.0
                    s11_x, s11_y = 356.0, 95.0
                    s_hip_x, s_hip_y = 204.0, 690.0
                
                # 실시간 웹캠 및 미디어파이프 AR 오버레이 HTML 컴포넌트 구조화 정의
                ar_html = f"""
                <div id="canvas-container">
                    <video id="webcam" width="640" height="480" autoplay playsinline style="display:none;"></video>
                    <canvas id="ar-canvas" width="640" height="480"></canvas>
                </div>
                
                <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
                <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js" crossorigin="anonymous"></script>
                
                <script>
                    // 최소제곱법 기반 3점 유사 변환 Solver (가로세로 비율 고정)
                    function getSimilarityTransform(src, dst) {{
                        let AtA = [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]
                        ];
                        let AtB = [0, 0, 0, 0];
                        
                        for (let i = 0; i < 3; i++) {{
                            const x = src[i].x, y = src[i].y;
                            const u = dst[i].x, v = dst[i].y;
                            
                            AtA[0][0] += x*x + y*y;
                            AtA[0][2] += x;
                            AtA[0][3] += y;
                            
                            AtA[1][1] += x*x + y*y;
                            AtA[1][2] += -y;
                            AtA[1][3] += x;
                            
                            AtA[2][2] += 1;
                            
                            AtA[3][3] += 1;
                            
                            AtB[0] += x*u + y*v;
                            AtB[1] += -y*u + x*v;
                            AtB[2] += u;
                            AtB[3] += v;
                        }}
                        
                        AtA[1][0] = AtA[0][1];
                        AtA[2][0] = AtA[0][2];
                        AtA[2][1] = AtA[1][2];
                        AtA[3][0] = AtA[0][3];
                        AtA[3][1] = AtA[1][3];
                        AtA[3][2] = AtA[2][3];
                        
                        let M = [];
                        for (let i = 0; i < 4; i++) {{
                            M[i] = [AtA[i][0], AtA[i][1], AtA[i][2], AtA[i][3], AtB[i]];
                        }}
                        
                        for (let i = 0; i < 4; i++) {{
                            let maxEl = Math.abs(M[i][i]);
                            let maxRow = i;
                            for (let k = i+1; k < 4; k++) {{
                                if (Math.abs(M[k][i]) > maxEl) {{
                                    maxEl = Math.abs(M[k][i]);
                                    maxRow = k;
                                }}
                            }}
                            
                            for (let k = i; k < 5; k++) {{
                                let tmp = M[maxRow][k];
                                M[maxRow][k] = M[i][k];
                                M[i][k] = tmp;
                            }}
                            
                            if (Math.abs(M[i][i]) < 1e-6) {{
                                return [1, 0, 0, 1, 0, 0];
                            }}
                            
                            for (let k = i+1; k < 4; k++) {{
                                let c = -M[k][i] / M[i][i];
                                for (let j = i; j < 5; j++) {{
                                    if (i === j) {{
                                        M[k][j] = 0;
                                    }} else {{
                                        M[k][j] += c * M[i][j];
                                    }}
                                }}
                            }}
                        }}
                        
                        let P = [0, 0, 0, 0];
                        for (let i = 3; i >= 0; i--) {{
                            P[i] = M[i][4] / M[i][i];
                            for (let k = i-1; k >= 0; k--) {{
                                M[k][4] -= M[k][i] * P[i];
                            }}
                        }}
                        
                        const a = P[0];
                        const b = P[1];
                        const tx = P[2];
                        const ty = P[3];
                        
                        return [a, b, -b, a, tx, ty];
                    }}
                    
                    const videoElement = document.getElementById('webcam');
                    const canvasElement = document.getElementById('ar-canvas');
                    const canvasCtx = canvasElement.getContext('2d');
                    
                    const imgTorso = new Image();
                    imgTorso.src = "{base64_torso}";
                    
                    const imgHead = new Image();
                    let hasHead = false;
                    if ("{base64_head}" !== "") {{
                        imgHead.src = "{base64_head}";
                        hasHead = true;
                    }}
                    
                    const torsoMeta = {{
                        "skeleton": {{ s_w: 300, s_h: 415 }},
                        "muscles": {{ s_w: 300, s_h: 415 }},
                        "nervous": {{ s_w: 300, s_h: 415 }},
                        "circulatory": {{ s_w: 300, s_h: 415 }},
                        "digestive": {{ s_w: 300, s_h: 415 }},
                        "respiratory": {{ s_w: 300, s_h: 415 }},
                        "excretory": {{ s_w: 300, s_h: 415 }}
                    }};
                    
                    const activeOrgan = "{active_organ}";
                    
                    function onResults(results) {{
                        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                        
                        canvasCtx.save();
                        canvasCtx.translate(canvasElement.width, 0);
                        canvasCtx.scale(-1, 1);
                        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
                        canvasCtx.restore();
                        
                        if (results.poseLandmarks) {{
                            const lms = results.poseLandmarks;
                            const w = canvasElement.width;
                            const h = canvasElement.height;
                            
                            // 11번(좌측어깨) 또는 12번(우측어깨) 랜드마크가 누락되면 드로잉 스킵하여 에러 차단
                            if (!lms[11] || !lms[12]) {{
                                return;
                            }}
                            
                            const p11 = lms[11], p12 = lms[12];
                            const p23 = lms[23] || null;
                            const p24 = lms[24] || null;
                            
                            const rs_x = w - (p12.x * w);
                            const rs_y = p12.y * h;
                            const ls_x = w - (p11.x * w);
                            const ls_y = p11.y * h;
                            
                            let hx = 0;
                            let hy = 0;
                            if (p23 && p24) {{
                                const rh_x = w - (p24.x * w);
                                const rh_y = p24.y * h;
                                const lh_x = w - (p23.x * w);
                                const lh_y = p23.y * h;
                                hx = (rh_x + lh_x) / 2.0;
                                hy = (rh_y + lh_y) / 2.0;
                            }}
                            
                            // 1. 몸통 정밀 유사 변환 3점 매핑
                            const cx = (rs_x + ls_x) / 2.0;
                            const cy = (rs_y + ls_y) / 2.0;
                            
                            const dx = ls_x - rs_x;
                            const dy = ls_y - rs_y;
                            const actualWidth = Math.sqrt(dx*dx + dy*dy);
                            
                            // 골반(23, 24번) 가시성 저하 또는 화면 하단 경계 이탈 시 가상 골반 중심 보간
                            const isHipInvalid = (!p23 || !p24 || p23.visibility < 0.55 || p24.visibility < 0.55 || p23.y > 0.92 || p24.y > 0.92);
                            if (isHipInvalid) {{
                                const sh_left_x = Math.min(rs_x, ls_x);
                                const sh_right_x = Math.max(rs_x, ls_x);
                                const sh_left_y = (rs_x < ls_x) ? rs_y : ls_y;
                                const sh_right_y = (rs_x < ls_x) ? ls_y : rs_y;
                                
                                const v_dx = sh_right_x - sh_left_x;
                                const v_dy = sh_right_y - sh_left_y;
                                const v_actual_w = Math.sqrt(v_dx*v_dx + v_dy*v_dy);
                                
                                const ux_down = -v_dy / (v_actual_w + 1e-5);
                                const uy_down = v_dx / (v_actual_w + 1e-5);
                                const virtualHeight = v_actual_w * 1.38;
                                hx = cx + ux_down * virtualHeight;
                                hy = cy + uy_down * virtualHeight;
                            }}
                            
                            const meta = torsoMeta[activeOrgan];
                            if (meta && imgTorso.complete) {{
                                // 3점 유사 변환 행렬 연산
                                // 에셋 소스 3점: 우측어깨(52, 95), 좌측어깨(356, 95), 골반중심(204, 690)
                                // 타깃 3점: rs_x/y, ls_x/y, hx/hy
                                const transformMat = getSimilarityTransform(
                                    [{{x: 800.0 - {s12_x}, y: {s12_y}}}, {{x: 800.0 - {s11_x}, y: {s11_y}}}, {{x: 800.0 - {s_hip_x}, y: {s_hip_y}}}],
                                    [{{x: rs_x, y: rs_y}}, {{x: ls_x, y: ls_y}}, {{x: hx, y: hy}}]
                                );
                                
                                const a = transformMat[0];
                                const b = transformMat[1];
                                const tx = transformMat[4];
                                const ty = transformMat[5];
                                
                                canvasCtx.save();
                                canvasCtx.transform(-a, -b, -b, a, a * 800.0 + tx, b * 800.0 + ty);
                                canvasCtx.globalAlpha = 0.85;
                                canvasCtx.drawImage(imgTorso, 0, 0);
                                canvasCtx.restore();
                            }}
                            
                            // 2. 머리 정밀 유사 변환 3점 매핑 (배설계 제외)
                            if (hasHead && imgHead.complete) {{
                                const p3 = lms[3]; // left eye outer
                                const p6 = lms[6]; // right eye outer
                                const p9 = lms[9]; // mouth left
                                const p10 = lms[10]; // mouth right
                                
                                // 이목구비 랜드마크가 다 존재할 때만 안전하게 그리기
                                if (p3 && p6 && p9 && p10) {{
                                    const re_out_x = w - (p6.x * w);
                                    const re_out_y = p6.y * h;
                                    const le_out_x = w - (p3.x * w);
                                    const le_out_y = p3.y * h;
                                    
                                    const m9_x = w - (p10.x * w);
                                    const m9_y = p10.y * h;
                                    const m10_x = w - (p9.x * w);
                                    const m10_y = p9.y * h;
                                    const mouth_cx = (m9_x + m10_x) / 2.0;
                                    const mouth_cy = (m9_y + m10_y) / 2.0;
                                    
                                    // 3점 유사 변환 행렬 연산 (머리용)
                                    // 에셋 소스 3점: 우안외측(s6_hx, s6_hy), 좌안외측(s3_hx, s3_hy), 입중심(s_mouth_hx, s_mouth_hy)
                                    // 타깃 3점: re_out, le_out, mouth_cx/cy
                                    const headTransformMat = getSimilarityTransform(
                                        [{{x: 262.0 - {s6_hx}, y: {s6_hy}}}, {{x: 262.0 - {s3_hx}, y: {s3_hy}}}, {{x: 262.0 - {s_mouth_hx}, y: {s_mouth_hy}}}],
                                        [{{x: re_out_x, y: re_out_y}}, {{x: le_out_x, y: le_out_y}}, {{x: mouth_cx, y: mouth_cy}}]
                                    );
                                    
                                    const headScale = 0.75;
                                    const ha = headTransformMat[0] * headScale;
                                    const hb = headTransformMat[1] * headScale;
                                    const tcx = (re_out_x + le_out_x + mouth_cx) / 3.0;
                                    const tcy = (re_out_y + le_out_y + mouth_cy) / 3.0;
                                    const htx = headTransformMat[4] * headScale + tcx * (1.0 - headScale);
                                    const hty = headTransformMat[5] * headScale + tcy * (1.0 - headScale) - 15.0;
                                    
                                    canvasCtx.save();
                                    canvasCtx.transform(-ha, -hb, -hb, ha, ha * 262.0 + htx, hb * 262.0 + hty);
                                    canvasCtx.globalAlpha = 0.85;
                                    canvasCtx.drawImage(imgHead, 0, 0);
                                    canvasCtx.restore();
                                }}
                            }}
                        }}
                    }}
                    
                    const pose = new Pose({{
                        locateFile: (file) => {{
                            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${{file}}`;
                        }}
                    }});
                    
                    pose.setOptions({{
                        modelComplexity: 1,
                        smoothLandmarks: true,
                        minDetectionConfidence: 0.5,
                        minTrackingConfidence: 0.5
                    }});
                    
                    pose.onResults(onResults);
                    
                    const camera = new Camera(videoElement, {{
                        onFrame: async () => {{
                            await pose.send({{image: videoElement}});
                        }},
                        width: 640,
                        height: 480
                    }});
                    camera.start();
                </script>
                """
                # 브라우저 강한 iframe 캐싱을 무력화하기 위한 Cache-Busting 타임스탬프 주석 추가 주입
                import time
                ar_html_unique = ar_html + f"\n<!-- Rerun-ID: {time.time()} -->"
                st.components.v1.html(ar_html_unique, height=520)
            else:
                # 아무 기관도 활성화되지 않은 디폴트 웹캠 피드
                ar_html_empty = """
                <div id="canvas-container">
                    <video id="webcam" width="640" height="480" autoplay playsinline style="display:none;"></video>
                    <canvas id="ar-canvas" width="640" height="480"></canvas>
                </div>
                <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
                <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js" crossorigin="anonymous"></script>
                <script>
                    const videoElement = document.getElementById('webcam');
                    const canvasElement = document.getElementById('ar-canvas');
                    const canvasCtx = canvasElement.getContext('2d');
                    
                    function onResults(results) {
                        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                        canvasCtx.save();
                        canvasCtx.translate(canvasElement.width, 0);
                        canvasCtx.scale(-1, 1);
                        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
                        canvasCtx.restore();
                    }
                    
                    const pose = new Pose({
                        locateFile: (file) => {
                            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
                        }
                    });
                    pose.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
                    pose.onResults(onResults);
                    
                    const camera = new Camera(videoElement, {
                        onFrame: async () => {
                            await pose.send({image: videoElement});
                        },
                        width: 640,
                        height: 480
                    });
                    camera.start();
                </script>
                """
                import time
                ar_html_empty_unique = ar_html_empty + f"\n<!-- Rerun-ID: {time.time()} -->"
                st.components.v1.html(ar_html_empty_unique, height=520)



