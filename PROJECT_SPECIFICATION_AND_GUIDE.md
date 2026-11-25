# mmWave FMCW Radar Simulation & 3D Digital Twin Platform
## Comprehensive Technical Specification, Scientific Foundations & Execution Guide
## مواصفات المشروع التقنية والأسس الرياضية ودليل التشغيل الشامل (عربي / English)

---

## Table of Contents / جدول المحتويات
1. [English Section](#english-section)
   - [1. Executive Summary & Architecture](#1-executive-summary--system-architecture)
   - [2. Mathematical & Physical Principles](#2-mathematical--physical-principles)
     - [2.1 FMCW Signal & Dechirp Mixing](#21-fmcw-signal--dechirp-mixing)
     - [2.2 TDM-MIMO Virtual Array & Doppler Compensation](#22-tdm-mimo-virtual-array--doppler-compensation)
     - [2.3 Geometric Optics & ITU-R P.2040-3 Propagation](#23-geometric-optics--itu-r-p2040-3-wave-propagation)
     - [2.4 Hardware & Channel Impairments](#24-hardware--channel-impairments)
     - [2.5 Radar Signal Processing (DSP) Pipeline](#25-radar-signal-processing-dsp-pipeline)
   - [3. Dashboard & Visualization Architecture](#3-dashboard--visualization-architecture)
   - [4. Execution & User Guide](#4-execution--user-guide)
   - [5. Configuration & Extension Guide](#5-configuration--extension-guide)
2. [القسم العربي (Arabic Section)](#القسم-العربي-arabic-section)
   - [1. الملخص التنفيذي وبنية النظام](#1-الملخص-التنفيذي-وبنية-النظام)
   - [2. الأسس الفيزيائية والرياضية للمحاكاة](#2-الأسس-الفيزيائية-والرياضية-للمحاكاة)
     - [2.1 إشارات الرادار النبضي المتصل (FMCW) وفك التضمين](#21-إشارات-الرادار-النبضي-المتصل-fmcw-وفك-التضمين)
     - [2.2 مصفوفة الهوائيات الافتراضية TDM-MIMO وتعويض الدوبلر](#22-مصفوفة-الهوائيات-الافتراضية-tdm-mimo-وتعويض-الدوبلر)
     - [2.3 انتشار الموجات الكهرومغناطيسية وفق ITU-R P.2040-3](#23-انتشار-الموجات-الكهرومغناطيسية-وفق-itu-r-p2040-3)
     - [2.4 النمذجة الدقيقة للضوضاء وتشوهات الأجهزة](#24-النمذجة-الدقيقة-للضوضاء-وتشوهات-الأجهزة)
     - [2.5 معالجة الإشارات الرادارية وخوارزميات كشف السقوط](#25-معالجة-الإشارات-الرادارية-وخوارزميات-كشف-السقوط)
   - [3. واجهة المستخدم الرسومية واللوحات التفاعلية](#3-واجهة-المستخدم-الرسومية-واللوحات-التفاعلية)
   - [4. دليل التشغيل خطوة بخطوة للفرق والمطورين](#4-دليل-التشغيل-خطوة-بخطوة-للفرق-والمطورين)
   - [5. دليل التعديل والتطوير والإضافة](#5-دليل-التعديل-والتطوير-والإضافة)

---

# English Section

## 1. Executive Summary & System Architecture

This project is a high-fidelity, physically principled simulation of a 60–77 GHz Millimeter-Wave (mmWave) Frequency-Modulated Continuous-Wave (FMCW) Time-Division Multiplexed Multiple-Input Multiple-Output (TDM-MIMO) radar operating inside an indoor multipath environment.

The system is strictly divided into six decoupled, reproducible pipeline stages:
1. **Scene Engine (`src/radarsim/scene`)**: 3D spatial geometry, room boundary mesh, furniture bounding volumes, and BVH-accelerated ray intersection.
2. **Motion Engine (`src/radarsim/motion`)**: Biomechanical skeletal kinematics, skinned 3D human avatar mesh deformation, forward kinematics, and activity state machines (walking, sitting, bending, falling).
3. **RF Propagation Engine (`src/radarsim/rf`)**: Geometric-optics ray tracing for Direct Line-of-Sight (LOS), specular reflections (image source method), diffuse scattering (Lambertian model), and wall penetration using ITU-R P.2040-3 complex dielectric parameters.
4. **Radar Frontend & ADC (`src/radarsim/radar`)**: Physics-level FMCW linear chirp generation, dechirp mixer, TDM chronological chirp scheduling, and independently toggleable hardware impairments (thermal noise, phase noise, IQ imbalance, ADC quantization/clipping, mutual coupling).
5. **Radar-Only DSP Pipeline (`src/radarsim/dsp`)**: Coherent Range FFT, dual slow-time processing (clutter suppression branch + stationary slow-time preservation), TDM Doppler phase shift compensation, Doppler FFT, 2D OS-CFAR detection, angle estimation (virtual array beamforming), DBSCAN clustering, GNN/Kalman centroid tracking, and rule-based fall detection.
6. **Streaming & 3D Dashboard (`src/radarsim/viz` & `viz/web`)**: FastAPI/WebSocket frame server streaming synchronized telemetry at 10–20 FPS to a responsive React 19 + Three.js 4-panel dashboard.

---

## 2. Mathematical & Physical Principles

### 2.1 FMCW Signal & Dechirp Mixing
The transmitted FMCW linear chirp signal for chirp index $m$ is defined as:
$$s_{\text{tx}}(t) = A_{\text{tx}} \exp\left(j 2\pi \left(f_c t + \frac{1}{2} S t^2\right) + \phi_0\right)$$
where $f_c$ is the carrier frequency (e.g., $60.75\text{ GHz}$), $S = \frac{B}{T_c}$ is the chirp slope, $B$ is sweep bandwidth, and $T_c$ is chirp duration.

A target situated at round-trip propagation delay $\tau(t) = \frac{2(R_0 + v_r t)}{c}$ reflects an attenuated, Doppler-shifted signal:
$$s_{\text{rx}}(t) = \sum_p A_p \cdot s_{\text{tx}}(t - \tau_p)$$

The dechirp mixer multiplies the received signal by the conjugate of the transmitted chirp $s_{\text{tx}}^*(t)$, producing the complex intermediate frequency (IF) beat signal:
$$s_{\text{IF}}(t) = s_{\text{rx}}(t) \cdot s_{\text{tx}}^*(t) \approx \sum_p A_p \exp\left(-j 2\pi \left(S \tau_p t + f_c \tau_p\right)\right)$$
- **Beat frequency:** $f_b = S \cdot \tau = \frac{2 S R}{c}$
- **Target range:** $R = \frac{c \cdot f_b}{2 S}$
- **Range resolution:** $\Delta R = \frac{c}{2 B}$

### 2.2 TDM-MIMO Virtual Array & Doppler Compensation
Using $N_{\text{tx}}$ transmitters and $N_{\text{rx}}$ receivers pulsed sequentially in time, the system synthesizes a virtual array of $N_{\text{virt}} = N_{\text{tx}} \times N_{\text{rx}}$ elements.

Because transmitters fire sequentially with inter-chirp interval $T_c$, a moving target with radial velocity $v_r$ induces an inter-TX Doppler phase error between consecutive TX pulses:
$$\Delta \phi_{\text{TDM}} = 2\pi f_c \frac{2 v_r T_c}{c} = 4\pi \frac{v_r T_c}{\lambda}$$

The DSP pipeline measures $v_r$ from the Doppler spectrum and performs coherent phase rotation correction prior to spatial angle estimation:
$$\tilde{X}_{\text{virt}}[m, \text{tx}, \text{rx}] = X[m, \text{tx}, \text{rx}] \cdot \exp\left(-j 4\pi \frac{v_r \cdot \text{tx} \cdot T_c}{\lambda}\right)$$

### 2.3 Geometric Optics & ITU-R P.2040-3 Wave Propagation
Propagation paths are evaluated using high-frequency asymptotic ray tracing:
1. **Free-Space Path Loss (Friis equation):**
   $$P_{\text{rx}} = P_{\text{tx}} \frac{G_{\text{tx}} G_{\text{rx}} \lambda^2}{(4\pi)^3 R_1^2 R_2^2} \sigma$$
2. **Dielectric Interaction (ITU-R P.2040-3):**
   The relative complex permittivity of building materials is calculated as:
   $$\eta = \epsilon_r - j \frac{\sigma_{\text{cond}}}{2\pi f \epsilon_0}$$
   Fresnel reflection coefficients for perpendicular (TE) and parallel (TM) polarizations:
   $$\Gamma_{\perp} = \frac{\cos\theta - \sqrt{\eta - \sin^2\theta}}{\cos\theta + \sqrt{\eta - \sin^2\theta}}, \quad \Gamma_{\parallel} = \frac{\eta \cos\theta - \sqrt{\eta - \sin^2\theta}}{\eta \cos\theta + \sqrt{\eta - \sin^2\theta}}$$
3. **Diffuse Scattering:** Modeled via Lambertian power distribution scaled by surface roughness Rayleigh parameter $\sigma_h$:
   $$\rho_{\text{diffuse}} = \exp\left(-8\pi^2 \left(\frac{\sigma_h \cos\theta}{\lambda}\right)^2\right)$$

### 2.4 Hardware & Channel Impairments
The radar frontend models ten independently toggleable hardware impairments:
- **Thermal Noise:** Derived explicitly from Johnson–Nyquist formula $P_n = k_B T B F$, where $k_B$ is Boltzmann's constant, $T=290\text{ K}$, $B$ is ADC bandwidth, and $F$ is receiver noise figure. Converted to ADC LSB units via documented system impedance ($50\ \Omega$) and ADC full-scale voltage ($1.0\text{ V}_{\text{pk}}$).
- **Phase Noise:** Injected into slow-time and fast-time chirps via Wiener process phase drift.
- **IQ Imbalance:** Modeled with amplitude mismatch $\epsilon$ and phase orthogonality error $\Delta \phi$:
  $$s_{\text{imb}}(t) = \text{Re}\{s(t)\} (1 + \epsilon/2) + j \left(\text{Im}\{s(t)\} (1 - \epsilon/2) + \text{Re}\{s(t)\} \sin\Delta\phi\right)$$
- **ADC Quantization & Saturation:** Configurable 12-bit / 14-bit / 16-bit uniform quantization with hard clipping at $\pm V_{\text{fs}}$.
- **Mutual Coupling & TX Leakage:** Direct antenna cross-talk modeled as complex S-parameter matrix leakage added before the LNA.

### 2.5 Radar Signal Processing (DSP) Pipeline
1. **Range FFT:** Applied along ADC fast-time samples with coherent gain normalization and windowing (Blackman/Hanning) to control sidelobes.
2. **Dual-Branch Clutter Processing:**
   - *Moving Target Branch:* Two-pulse or multi-pulse high-pass MTI filter removes stationary walls and furniture.
   - *Static Slow-Time Branch:* Preserved untouched to allow stationary human presence detection and micro-breathing estimation.
3. **Doppler FFT:** Computes radial velocity spectrum across chirp loops.
4. **2D OS-CFAR Detector:** Ordered-Statistic Constant False Alarm Rate filter computes adaptive noise threshold:
   $$V_{\text{th}} = \alpha \cdot Z_{(k)}$$
   where $Z_{(k)}$ is the $k$-th ordered sample among reference cells, maintaining robust detection under multi-target masking.
5. **Angle Estimation:** Computes 3D Azimuth and Elevation via FFT / Bartlett beamforming across the virtual array.
6. **Clustering & Tracking:** DBSCAN spatial clustering associates radar detections; a Kalman filter tracks target 3D centroid, height, and vertical velocity.
7. **Rule-Based Fall Detection:** Evaluates temporal features:
   - Rapid negative vertical velocity ($\Delta z / \Delta t < -1.5\text{ m/s}$).
   - Significant drop in track centroid height ($z < 0.45\text{ m}$).
   - Post-fall immobility / stillness duration ($> 1.0\text{ s}$).
   - Normal sitting, kneeling, and lying down serve as hard negatives to avoid false alarms.

---

## 3. Dashboard & Visualization Architecture

The web interface (`viz/web`) delivers a real-time synchronized 4-panel dashboard:
- **Panel 1 (Ground Truth 3D Scene):** Interactive Three.js viewport showing the architectural room, textured walls, furniture, and animated human mesh. Supports full mouse orbit, pan, and zoom.
- **Panel 2 (Geometric-Optics RF Waves):** Color-coded dynamic 3D ray trajectories (LOS, specular reflections, diffuse scattering, transmitted rays) fading with distance and attenuation.
- **Panel 3 (Radar-Derived 3D Observations):** Displays pure radar point clouds (colored by Doppler velocity), tracked bounding boxes, and sparse occupancy voxels. Ground-truth avatar overlay is toggled OFF by default to guarantee zero-leakage radar validation.
- **Panel 4 (Radar Analytics & Diagnostics):** Real-time 2D Range-Doppler Spectrogram Heatmap, Range-Azimuth B-scan / Micro-Doppler waterfall, height & vertical velocity tracking graphs, and fall alarm notifications.

---

## 4. Execution & User Guide

### 4.1 Prerequisites
- **Python**: 3.10, 3.11, or 3.12
- **Node.js**: v18.0 or newer (includes `npm`)
- **Git**

### 4.2 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/amr-sokkar/radar-simulation-V1.git
   cd radar-simulation-V1
   ```
2. Set up the Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate

   pip install -e .
   ```
3. Set up the frontend dependencies:
   ```bash
   cd viz/web
   npm install
   cd ../..
   ```

### 4.3 Running the Simulation
#### Method A: Using Automated Scripts (Windows)
Open two PowerShell terminals in the repository root:
- **Terminal 1 (Backend):**
  ```powershell
  .\run_backend.ps1
  ```
- **Terminal 2 (Frontend):**
  ```powershell
  .\run_frontend.ps1
  ```

#### Method B: Manual Command Line (All Platforms)
- **Terminal 1 (FastAPI Backend Server on Port 8000):**
  ```bash
  python -m uvicorn radarsim.viz.server:create_app --factory --host 127.0.0.1 --port 8000 --reload
  ```
- **Terminal 2 (Vite Development Server on Port 5173):**
  ```bash
  cd viz/web
  npm run dev
  ```
Open your browser at: **`http://localhost:5173/`**

### 4.4 Automated Testing & Verification
Verify full physical and analytical integrity with:
```bash
# Validate configurations against analytical radar formulas
python scripts/validate_config.py

# Run the complete test suite (physics, kinematics, RF, radar, DSP, integration)
pytest -v
```

---

## 5. Configuration & Extension Guide

| File | Purpose | Key Parameters to Modify |
|---|---|---|
| `configs/radar_config.yaml` | Radar RF frontend | Carrier frequency ($f_c$), sweep bandwidth ($B$), chirp slope ($S$), ADC sample rate ($F_s$), antenna array geometry. |
| `configs/scene.json` | Room and scene layout | Room length/width/height, wall materials, furniture dimensions, positions, and orientations. |
| `configs/material_db.yaml` | ITU dielectric parameters | Material permittivity ($\epsilon_r$), conductivity ($\sigma$), surface roughness ($\sigma_h$). |
| `configs/motion_config.yaml` | Avatar motion & kinematics | Walk trajectory waypoints, speed, fall profile, action state sequence. |

---
---

# القسم العربي (Arabic Section)

## 1. الملخص التنفيذي وبنية النظام

يمثل هذا المشروع منصة محاكاة فيزيائية متقدمة وعالية الدقة لرادارات الموجات المليمترية (mmWave) من طراز FMCW (رادار الموجة المتصلة معدلة التردد) العاملة بتقنية التجميع الزمني للمدخلات والمخرجات المتعددة (TDM-MIMO) في نطاق الترددات 60 إلى 77 جيجاهرتز، والمخصصة للبيئات الداخلية متعددة المسارات (Multipath Indoor Environments) ومراقبة الحركة البشرية وكشف السقوط بدقة.

تتكون بنية النظام البرمجية من ست مراحل مفصولة ومصممة بأعلى معايير النزاهة الهندسية:
1. **محرك البيئة والمكان (`src/radarsim/scene`)**: يمثل الهندسة الفراغية ثلاثية الأبعاد للغرفة، طبقات الجدران، وكتل الأثاث مع تسريع تقاطعات الأشعة الفراغية باستخدام هياكل BVH.
2. **محرك الحركة الحركية (`src/radarsim/motion`)**: محاكاة الميكانيكا الحيوية للهيكل العظمي البشري، تشويه الشبكة ثلاثية الأبعاد (Skinning)، ومخطط الحالات السلوكية (مشي، جلوس، انحناء، سقوط، استلقاء).
3. **محرك انتشار الموجات الكهرومغناطيسية (`src/radarsim/rf`)**: تتبع الأشعة وفق البصريات الهندسية (Geometric Optics) لحساب خط الرؤية المباشر (LOS)، الانعكاسات المرآوية (Specular)، التشتت غير المنتظم (Diffuse Scattering)، والنفاذ عبر الجدران وفق توصية الاتحاد الدولي للاتصالات ITU-R P.2040-3.
4. **واجهة الرادار ومحول الإشارة التناظرية للرقمية (`src/radarsim/radar`)**: توليد النبضات الخطية (Chirps)، نموذج خلط الإشارات وإزالة التضمين (Dechirp Mixer)، جدولة TDM الزمنية، ونمذجة شاملة لعيوب ومؤثرات العتاد (الضوضاء الحرارية، ضوضاء الطور، اختلال قنوات IQ، وتكميم الـ ADC).
5. **معالجة الإشارات الرادارية الصافية (`src/radarsim/dsp`)**: تحويل فورييه للمدى (Range FFT)، مسار فصل الأهداف المتحركة مع الحفاظ على الأهداف الثابتة، تعويض إزاحة طور دوبلر لمصفوفة TDM، تحويل فورييه للسرعة (Doppler FFT)، كاشف OS-CFAR ثنائي الأبعاد، تقدير الزوايا، التجميع العنقودي، التتبع المرشح (Kalman Tracker)، وخوارزمية كشف السقوط القائمة على القواعد الفيزيائية.
6. **خادم البث وواجهة الويب ثلاثية الأبعاد (`src/radarsim/viz` & `viz/web`)**: خادم متزامن عبر بروتوكول WebSocket ينقل الإطارات بمعدل 10 إلى 20 إطاراً في الثانية إلى لوحة قيادة تفاعلية مدمجة مبنية باستخدام React 19 و Three.js مقسمة إلى 4 شاشات متزامنة لحظياً.

---

## 2. الأسس الفيزيائية والرياضية للمحاكاة

### 2.1 إشارات الرادار النبضي المتصل (FMCW) وفك التضمين
تُعرّف إشارة النبضة المرسلة الخطية (Chirp) رياضياً بالعلاقة:
$$s_{\text{tx}}(t) = A_{\text{tx}} \exp\left(j 2\pi \left(f_c t + \frac{1}{2} S t^2\right) + \phi_0\right)$$
حيث $f_c$ هو التردد الحامل (مثلاً $60.75\text{ GHz}$)، و $S = \frac{B}{T_c}$ هو ميل النبضة (Chirp Slope)، و $B$ هو عرض النطاق الترددي الممسوح، و $T_c$ هو زمن النبضة الواحدة.

عند انعكاس الإشارة عن هدف يقع على بعد يتغير مع الزمن بمقدار تأخير زمني ذهاباً وإياباً $\tau(t) = \frac{2(R_0 + v_r t)}{c}$، تكون الإشارة المستقبلة:
$$s_{\text{rx}}(t) = \sum_p A_p \cdot s_{\text{tx}}(t - \tau_p)$$

يقوم المازج (Dechirp Mixer) بضرب الإشارة المستقبلة بالمرافق المعقد للإشارة المرسلة $s_{\text{tx}}^*(t)$، لإنتاج إشارة التردد البيني (IF Beat Signal):
$$s_{\text{IF}}(t) = s_{\text{rx}}(t) \cdot s_{\text{tx}}^*(t) \approx \sum_p A_p \exp\left(-j 2\pi \left(S \tau_p t + f_c \tau_p\right)\right)$$
- **تردد النبضة البيني (Beat Frequency):** $f_b = S \cdot \tau = \frac{2 S R}{c}$
- **حساب المدى (Target Range):** $R = \frac{c \cdot f_b}{2 S}$
- **دقة الفصل المكاني في المدى (Range Resolution):** $\Delta R = \frac{c}{2 B}$

### 2.2 مصفوفة الهوائيات الافتراضية TDM-MIMO وتعويض الدوبلر
باستخدام عدد $N_{\text{tx}}$ من هوائيات الإرسال و $N_{\text{rx}}$ من هوائيات الاستقبال تعمل بالتناوب الزمني، يبني النظام مصفوفة افتراضية مكافئة بحجم:
$$N_{\text{virt}} = N_{\text{tx}} \times N_{\text{rx}}$$

نظراً لإرسال نبضات الهوائيات بتتابع زمني بفاصل $T_c$، فإن أي حركة للهدف بسرعة قطرية $v_r$ تتسبب في إزاحة طور دوبلر إضافية غير مرغوبة بين نبضات هوائيات الإرسال المتعاقبة:
$$\Delta \phi_{\text{TDM}} = 4\pi \frac{v_r T_c}{\lambda}$$

يقوم نظام معالجة الإشارات DSP بقياس سرعة دوبلر للهدف وتعويض هذا الخطأ الطوري قبل البدء بتقدير الزوايا الفراغية:
$$\tilde{X}_{\text{virt}}[m, \text{tx}, \text{rx}] = X[m, \text{tx}, \text{rx}] \cdot \exp\left(-j 4\pi \frac{v_r \cdot \text{tx} \cdot T_c}{\lambda}\right)$$

### 2.3 انتشار الموجات الكهرومغناطيسية وفق ITU-R P.2040-3
يتم حساب طاقة ومسارات الموجات بدقة باستخدام البصريات الهندسية:
1. **معادلة انتقال فريس للمسار الحر:**
   $$P_{\text{rx}} = P_{\text{tx}} \frac{G_{\text{tx}} G_{\text{rx}} \lambda^2}{(4\pi)^3 R_1^2 R_2^2} \sigma$$
2. **الخصائص العازلة للمواد وفق المعيار الدولي ITU-R P.2040-3:**
   يتم حساب السماحية الكهربائية النسبية المعقدة لمواد البناء والأثاث عبر التردد:
   $$\eta = \epsilon_r - j \frac{\sigma_{\text{cond}}}{2\pi f \epsilon_0}$$
   ومعاملات فرينل للانعكاس للاستقطاب العمودي والأفقي:
   $$\Gamma_{\perp} = \frac{\cos\theta - \sqrt{\eta - \sin^2\theta}}{\cos\theta + \sqrt{\eta - \sin^2\theta}}, \quad \Gamma_{\parallel} = \frac{\eta \cos\theta - \sqrt{\eta - \sin^2\theta}}{\eta \cos\theta + \sqrt{\eta - \sin^2\theta}}$$
3. **التشتت اللامبيرتي السطحي (Diffuse Scattering):** يعتمد على عامل خشونة السطح لرايلي $\sigma_h$:
   $$\rho_{\text{diffuse}} = \exp\left(-8\pi^2 \left(\frac{\sigma_h \cos\theta}{\lambda}\right)^2\right)$$

### 2.4 النمذجة الدقيقة للضوضاء وتشوهات الأجهزة
تتضمن المحاكاة مفاتيح تحكم مستقلة لتمكين أو عزل العيوب الآتية:
- **الضوضاء الحرارية (Thermal Noise):** محسوبة فيزيائياً من صيغة جونسون-نايكويست $P_n = k_B T B F$ ومحولة بدقة إلى وحدات الـ ADC الرقمية بناءً على ممانعة النظام ($50\ \Omega$) والجهد الأقصى ($1.0\text{ V}_{\text{pk}}$).
- **ضوضاء الطور (Phase Noise):** تحاكي انزياح الطور العشوائي (Wiener process) عبر الزمن البطيء والسريع.
- **اختلال قنوات الـ IQ:** نمذجة دقيقة لعدم توازن السعة $\epsilon$ وخطأ التعامد الزاوي $\Delta\phi$.
- **تكميم وتسطيح الـ ADC:** محاكاة تكميم موحد قابل للضبط (12، 14، أو 16 بت) مع قطع حاد للإشارة عند تجاوز السعة القصوى.
- **التسريب بين الهوائيات (Mutual Coupling):** نمذجة الاقتران المباشر بين الهوائيات عبر مصفوفة معقدة لمعاملات التشتت S-parameters.

### 2.5 معالجة الإشارات الرادارية وخوارزميات كشف السقوط
1. **تحويل فورييه للمدى (Range FFT):** نافذة هانينج أو بلاكمان لتخفيف القمم الجانبية مع تسوية متماسكة للكسب.
2. **المعالجة الثنائية للكتم والركام:** فرع خاص لكتم الصدى الثابت (MTI) للأهداف المتحركة، وفرع موازٍ غير ممسوس للأهداف الثابتة لكشف التنفس والتواجد.
3. **تحويل فورييه للسرعة (Doppler FFT):** يحدد السرعات القطرية عبر النبضات.
4. **كاشف OS-CFAR المتكيف:** يستخدم خوارزمية إحصاء الترتيب لتحديد عتبة الضوضاء وتجنب حجب الأهداف المتجاورة.
5. **تقدير الزوايا ثلاثية الأبعاد:** تشكيل الحزمة الرقمية عبر المصفوفة الافتراضية لحساب زاوية السمت (Azimuth) وزاوية الارتفاع (Elevation).
6. **التجميع والتتبع الفراغي:** خوارزمية DBSCAN لتجميع النقاط، ومرشح كالمان لتقدير مركز الثقل وارتفاع الجسم وسرعته الرأسية.
7. **كشف السقوط الحركي الفيزيائي:**
   - سرعة هبوط رأسية حادة تتجاوز $-1.5\text{ m/s}$.
   - انخفاض سريع في ارتفاع مركز الثقل تحت $0.45\text{ m}$.
   - سكون وانعدام حركة لفترة تتجاوز ثانية واحدة بعد الارتطام.
   - تمييز حالات الجلوس الطبيعي، الانحناء، والركوع كحالات سلبية صعبة لمنع الإنذارات الخاطئة.

---

## 3. واجهة المستخدم الرسومية واللوحات التفاعلية

توفر الواجهة الرسومية المبنية بتقنيات الويب الحديثة لوحة تحكم مقسمة بنظام 2 × 2 متزامنة تماماً في إطار زمني موحد:
- **اللوحة الأولى (مشهد الواقع الأرضي ثلاثي الأبعاد):** عرض الغرفة والجدران والأثاث ومجسم بشري متحرك بكامل مفاصله. تتيح التحكم الكامل بالفأرة بالدوران والتقريب والتحريك.
- **اللوحة الثانية (مسارات الموجات الكهرومغناطيسية):** خطوط ملونة ثلاثية الأبعاد تمثل الأشعة المباشرة، الانعكاسات، التشتت، والنفاذ مع تدريج لوني بحسب شدة الفقد في الطاقة.
- **اللوحة الثالثة (ملاحظات الرادار الصافية ثلاثية الأبعاد):** ما "يراه" الرادار فقط بدون أي تسريب لبيانات الحقيقة الأرضية: سحابة نقاط ثلاثية الأبعاد ملونة بسرعات دوبلر، وصندوق التتبع المجسم، وحجوم الإشغال الفراغية (مع زر اختياري لعرض الحقيقة الأرضية للمقارنة).
- **اللوحة الرابعة (التحليلات والمخططات اللحظية):** خريطة حرارية ثنائية الأبعاد للمدى والسرعة (Range-Doppler Heatmap)، شلال الدوبلر المصغر (Micro-Doppler Waterfall)، منحنى الارتفاع والسرعة الرأسية مع الزمن، وشريط تنبيهات السقوط اللحظي.

---

## 4. دليل التشغيل خطوة بخطوة للفرق والمطورين

### 4.1 المتطلبات الأساسية
- **بايثون (Python)**: إصدار 3.10 أو 3.11 أو 3.12.
- **نود (Node.js)**: إصدار 18.0 أو أحدث (يتضمن `npm`).
- **أداة Git**.

### 4.2 التثبيت والإعداد
1. استنساخ المستودع من GitHub:
   ```bash
   git clone https://github.com/amr-sokkar/radar-simulation-V1.git
   cd radar-simulation-V1
   ```
2. إنشاء وتفعيل بيئة العمل الافتراضية لبايثون وتثبيت المكتبات:
   ```bash
   python -m venv .venv
   # لأنظمة ويندوز:
   .venv\Scripts\activate
   # لأنظمة لينكس وماك:
   source .venv/bin/activate

   pip install -e .
   ```
3. تثبيت حزم واجهة الويب:
   ```bash
   cd viz/web
   npm install
   cd ../..
   ```

### 4.3 تشغيل المحاكاة
#### الطريقة الأولى: عبر السكربتات المجهزة (أنظمة ويندوز)
افتح نافذتي PowerShell في المجلد الرئيسي للمشروع:
- **النافذة الأولى (الخادم الخلفي للبايثون):**
  ```powershell
  .\run_backend.ps1
  ```
- **النافذة الثانية (خادم الواجهة الرسومية):**
  ```powershell
  .\run_frontend.ps1
  ```

#### الطريقة الثانية: الأوامر اليدوية المباشرة (جميع الأنظمة)
- **النافذة الأولى (خادم بايثون عبر المنفذ 8000):**
  ```bash
  python -m uvicorn radarsim.viz.server:create_app --factory --host 127.0.0.1 --port 8000 --reload
  ```
- **النافذة الثانية (خادم الواجهة الأمامية عبر المنفذ 5173):**
  ```bash
  cd viz/web
  npm run dev
  ```
ثم افتح الرابط في المتصفح: **`http://localhost:5173/`**

### 4.4 الفحص والاختبار الآلي
للتأكد من صحة الحسابات الفيزيائية وعدم حدوث أي انكسار برمجي:
```bash
# 1. التحقق من صحة ملفات التكوين والرياضيات الفيزيائية
python scripts/validate_config.py

# 2. تشغيل حزمة الاختبارات الشاملة (68 اختباراً برمجياً وفيزيائياً)
pytest -v
```

---

## 5. دليل التعديل والتطوير والإضافة

يمكن لأي مطور في الفريق تعديل المنظومة وتطويرها بسهولة بفضل التصميم المنفصل:

| الملف أو المجلد | الوظيفة | ما يمكنك تعديله وتطويره |
|---|---|---|
| `configs/radar_config.yaml` | إعدادات الرادار العتادية | التردد الحامل، عرض النطاق الترددي، ميل النبضة، توزيع هوائيات الإرسال والاستقبال. |
| `configs/scene.json` | الهندسة الفراغية للغرفة | أبعاد الغرفة، مواقع وأحجام الجدران، الأبواب، والمكاتب والكراسي. |
| `configs/material_db.yaml` | قاعدة الخصائص الكهرومغناطيسية | قيم السماحية والتوصيلية وخشونة الأسطح للخرسانة، الخشب، والزجاج. |
| `configs/motion_config.yaml` | حركية وسلوك المجسم البشري | مسارات المشي، سرعة الحركة، نمط السقوط الحركي، وتوقيت الحالات. |
| `src/radarsim/rf/` | محرك الانتشار الكهرومغناطيسي | إضافة حيود الحواف (UTD)، أو تسريع تتبع الأشعة باستخدام كروت الشاشة GPU. |
| `src/radarsim/dsp/` | خوارزميات معالجة الإشارة | استبدال تشكيل الحزم بخوارزميات فائقة الدقة مثل MUSIC أو Capon، وإضافة شبكات الذكاء الاصطناعي لتصنيف الأنشطة. |
| `viz/web/src/` | واجهة المستخدم ثلاثية الأبعاد | تخصيص مظهر اللوحات، إضافة رسوم بيانية ومخططات تحليلية جديدة، أو تصدير البيانات بنقرة زر. |
