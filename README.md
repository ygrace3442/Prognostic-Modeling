# 👶 SAMSIN: AI-Based IVF Pregnancy Success Prediction

### (삼신: 난임 환자 임신 성공 여부 예측 솔루션)

> **"데이터의 진실을 통해 희망의 수치를 전하다."**
> **SAMSIN**은 25만 건의 난임 시술 데이터를 분석하여 임신 성공 확률을 예측하는 AI 솔루션입니다. 정형 데이터(Tabular)를 이미지로 변환하여 분석하는 **DeepInsight** 기법과 다중 모델 앙상블 전략을 통해 **AUC 0.7423**의 독보적인 정확도를 달성했습니다.

---

## 🏆 Project Achievement

* **Competition**: OZ Coding School x DACON AI Healthcare Hackathon
* **Team**: 6조 삼신할매와 아기동자
* **Result**: **Final AUC 0.7423 (Leaderboard Rank 1)**

---

## 🏗️ Model Architecture (Trinity Fusion)

우리는 단일 모델의 한계를 극복하기 위해 **"서로 다른 시각을 가진 3가지 엔진"**을 융합하는 **Trinity Fusion** 전략을 사용했습니다.

```mermaid
graph TD
    Data[Raw Clinical Data] --> Preprocessing[Advanced Preprocessing]
    Preprocessing --> A[Rank Breaker Model<br>(70%)]
    Preprocessing --> B[Hidden Card Model<br>(15%)]
    Preprocessing --> C[DeepInsight Model<br>(15%)]
    
    A -->|Tree + TabNet| Ensemble[Final Ensemble<br>Weighted Blending]
    B -->|AutoGluon Cleaned| Ensemble
    C -->|Tabular to Image + CNN| Ensemble
    
    Ensemble --> Final[Final Submission<br>AUC 0.7423]

```

### 1. Rank Breaker (The Commander)

* **Role**: 전체 예측 성능의 기반(Baseline) 확보
* **Tech**: XGBoost, LightGBM, CatBoost, TabNet
* **Detail**: 트리 모델과 정형 딥러닝 모델을 Stacking하여 안정적인 고득점을 유지.

### 2. Hidden Card (The Specialist)

* **Role**: 데이터 전처리 디테일 및 최적점 공략
* **Tech**: AutoGluon (Best Quality Preset)
* **Detail**: '회' 등 불필요한 문자열 제거, 정교한 결측치 처리, 통계적 이상치 제어를 통해 노이즈가 제거된 데이터 학습.

### 3. DeepInsight (The Visionary) 🌟 *Key Feature*

* **Role**: 비선형적 패턴 및 복잡한 상호작용 발견
* **Tech**: **t-SNE based Feature Mapping → CNN (ResNet18)**
* **Detail**: 엑셀 형태의 환자 데이터를 **이미지(Image)**로 변환하여, 수치로는 보이지 않는 환자 군집의 특성을 시각적으로 학습.

---

## 📊 Key Features & EDA

### 1. Data Cleaning ('The Purity')

* **Problem**: `총 시술 횟수` 등 주요 변수에 한글("1회", "2회 이상")과 숫자가 혼재.
* **Solution**: 정규표현식을 활용하여 비수치 데이터를 완벽하게 제거, 모델이 온전히 숫자에 집중하도록 처리.

### 2. Feature Engineering ('Medical Insight')

* **Efficiency Metrics**: 난자 채취 수 대비 이식 배아 수 비율 등 **생물학적 효율성** 지표 생성.
* **Age Binning**: 난임 시술 성공률이 급격히 꺾이는 연령 구간(35세, 38세, 40세 등)을 반영한 파생변수 생성.

---

## 🛠️ Installation & Usage

이 프로젝트는 4가지 핵심 모델 파이프라인을 하나의 `main_final_submission.py`로 통합했습니다.

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/samsin-project.git
cd samsin-project

# Install dependencies
pip install -r requirements.txt
pip install autogluon torch torchvision

```

### 2. Prepare Data

`open/` 폴더 내에 대회 데이터(`train.csv`, `test.csv`)를 위치시킵니다.

### 3. Run Pipeline

```bash
python main_final_submission.py

```

> **Note**: 전체 파이프라인 실행 시 GPU 환경(Colab Pro+ 권장)이 필요하며, 약 2~3시간이 소요될 수 있습니다.

---

## 💼 Business Value

**SAMSIN**은 단순한 예측 모델을 넘어, 난임 시장의 게임 체인저가 될 비즈니스 모델을 제안합니다.

| Model | Target | Value Proposition |
| --- | --- | --- |
| **B2B (SaaS)** | 난임 전문 병원 | 환자 상담용 AI 보조 솔루션 (구독형) |
| **B2C (Report)** | 난임 환자 | 68개 변수 분석 기반 개인 맞춤형 심층 리포트 |
| **B2G (Data)** | 지자체/정부 | 저출산 정책 수립을 위한 데이터 대시보드 |

---

## 👥 Team Members (6조)

* **김빛나 (Leader)**: EDA, Super Gap 파이프라인 설계, 총괄
* **김선화 (Engineer)**: AutoGluon 최적화, 전처리 디테일 강화 (Hidden Card)
* **양은서 (Modeling)**: DeepInsight 구현, 앙상블 전략 수립, Deep Learning
* **도금재 (Planning)**: 비즈니스 모델 설계, 발표 자료 작성

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

*For more details, please refer to the [Final Presentation PDF](https://www.google.com/search?q=./6%EC%A1%B0.%EC%82%BC%EC%8B%A0%ED%95%A0%EB%A7%A4%EC%99%80%EC%95%84%EA%B8%B0%EB%8F%99%EC%9E%90_%EC%B5%9C%EC%A2%85.pdf).*
