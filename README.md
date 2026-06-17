# NAVIGATION-GroundingDINO-

## 📱 UniGoal — 混合型店內導航應用

**UniGoal** 是一個AI驅動的零售店內導航應用，幫助顧客快速找到目標商品。該應用結合了GroundingDINO物體檢測、EasyOCR文字識別和本地VLM推理，無需雲端處理，確保隱私安全。

---

## 🎯 應用概述

| 項目 | 說明 |
|------|------|
| **使用場景** | 顧客在零售店內拍照並接收導航指引（例如："找到牛奶"、"找到麵包"） |
| **輸入方式** | 智慧型手機照片（無深度感測、無RGB-D、無已知位置） |
| **核心架構** | UniGoal基礎設施 + WMNav推理引擎 |
| **隱私保護** | 所有推理在本地進行，顧客照片不上傳雲端 |

---

## 🚀 使用流程說明

### 一、安裝與環境配置

#### 1. 前置要求
```bash
# 系統需求
- Python 3.8+
- CUDA 11.0+ (用於GPU推理，可選)
- 10GB+ 硬碟空間 (預訓練模型)
- 4GB+ RAM最小配置

# 軟體依賴
- FastAPI
- Ollama (本地VLM推理)
- PyTorch
- GroundingDINO
- EasyOCR
- NetworkX
```

#### 2. 克隆專案
```bash
git clone https://github.com/B1222017/NAVIGATION-GroundingDINO-.git
cd NAVIGATION-GroundingDINO-
```

#### 3. 建立虛擬環境
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

#### 4. 安裝依賴
```bash
pip install -r requirements.txt
```

#### 5. 下載模型
```bash
# 下載GroundingDINO預訓練權重
python scripts/download_models.py

# 啟動Ollama服務 (本地VLM推理)
ollama serve
# 在另一個終端拉取llama3.2-vision模型
ollama pull llama3.2-vision
```

---

### 二、地圖預建階段 (Batch Mode)

此階段在應用部署前進行，一次性預建店鋪的拓撲地圖。

#### 1. 準備店鋪環境照片
```
store_photos/
├── aisle_1.jpg
├── aisle_2.jpg
├── produce_section.jpg
├── dairy_section.jpg
└── frozen_section.jpg
```

**拍攝建議**：
- 每個區域拍2-3張照片 (不同角度)
- 照片解析度至少1080p
- 確保貨架簽牌清晰可見

#### 2. 執行批量地圖生成
```bash
python generate_topomap.py \
  --input_dir store_photos/ \
  --output_dir maps/ \
  --store_name my_store
```

**流程說明**：
1. **照片處理** → OCR讀取走道簽 (例如："走道7 — 乳製品")
2. **物體檢測** → GroundingDINO檢測每張照片的商品
3. **區域聚類** → Jaccard相似度聚類，生成店鋪區域 (乳製品、冷凍、生鮮等)
4. **圖結構生成** → NetworkX構建拓撲圖，節點=區域，邊=相鄰通路
5. **地圖渲染** → 視覺化結果儲存至 `maps/my_store/topomap.png`

**輸出檔案**：
```
maps/my_store/
├── topomap.pkl          # 序列化拓撲圖
├── topomap.png          # 地圖可視化
├── zone_info.json       # 各區域的商品清單
└── ocr_results.json     # 所有讀取的簽牌文本
```

---

### 三、運行時導航階段 (Per-Session)

顧客啟動應用後，逐步上傳照片接收導航指引。

#### 1. 啟動服務器
```bash
python -m server.server
```

**終端輸出**：
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

#### 2. 初始化會話
```bash
curl -X POST http://localhost:8000/session/init \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "my_store",
    "goal": "找到牛奶",
    "map_path": "maps/my_store/topomap.pkl"
  }'
```

**回應範例**：
```json
{
  "session_id": "sess_abc123",
  "goal": "找到牛奶",
  "status": "initialized",
  "current_node": "entrance",
  "instruction": "請拍攝您眼前的環境照片"
}
```

#### 3. 提交照片與接收導航

**步驟A** — 顧客拍攝並上傳照片：
```bash
curl -X POST http://localhost:8000/navigate \
  -H "Content-Type: multipart/form-data" \
  -F "session_id=sess_abc123" \
  -F "photo=@user_photo.jpg"
```

**步驟B** — 後端處理流程：

1. **目標分解** (Goal Decomposer)
   - 輸入："找到牛奶"
   - 輸出：檢測類別 = `["milk", "milk_bottle", "dairy_product", "refrigerated_shelf"]`

2. **照片分析** (物體檢測 + OCR)
   - GroundingDINO檢測上述類別
   - EasyOCR讀取走道簽
   - 結果：在照片中發現 "牛奶" (置信度0.85) 和 "走道7"

3. **好奇心評分** (Curiosity Scoring)
   - 根據已訪問區域，評估相鄰區域的可能性
   - 優先探索尚未確認的高機率區域

4. **VLM推理** (兩階段定位)
   - **階段1** (尋找走道)："您在走道7。乳製品區就在附近。"
   - **階段2** (精準定位)："請往左轉，在冷藏櫃上尋找。"

5. **ASK 動作** (可選)
   - 如果檢測不確定，詢問："您看到冷藏牛奶簽牌嗎？"

**回應範例**：
```json
{
  "session_id": "sess_abc123",
  "action": "NAVIGATE",
  "next_node": "dairy_aisle",
  "instruction": "直走15米，然後向左轉。您會看到乳製品區。",
  "confidence": 0.92,
  "detected_objects": [
    {"label": "milk_bottle", "confidence": 0.85, "location": "top_shelf"},
    {"label": "refrigerated_shelf", "confidence": 0.78}
  ],
  "ocr_text": ["Aisle 7", "Dairy Products"],
  "annotated_photo_url": "/results/annotated_sess_abc123.jpg"
}
```

#### 4. 到達確認 (User-Confirmed ARRIVED)

當顧客接近目標商品時：

```bash
curl -X POST http://localhost:8000/confirm_arrival \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sess_abc123",
    "photo": "@final_photo.jpg"
  }'
```

**後端驗證**：
1. 在照片中檢測目標商品 (牛奶)
2. 檢查位置信息 (例如：貨架高度、數量)
3. 等待顧客確認："是的，我找到了"

**最終回應**：
```json
{
  "status": "ARRIVED",
  "message": "恭喜！您已找到目標商品 - 牛奶。",
  "product_shelf_info": "冷藏櫃，中層",
  "nearby_products": ["yogurt", "cheese", "butter"],
  "session_duration": "2 min 34 sec"
}
```

---

## ⚙️ 配置參數

編輯 `server/config.py` 調整檢測精度：

```python
# 物體檢測閾值
BOX_THRESHOLD = 0.30          # GroundingDINO置信度 (0.0-1.0)
TEXT_THRESHOLD = 0.25         # OCR置信度
TOP_K = 5                     # 返回前K個檢測結果

# 自適應策略
USE_ADAPTIVE_THRESHOLD = True  # 啟用動態閾值調整
FALLBACK_THRESHOLD = 0.20      # 降級閾值 (未檢測到時)
CURIOSITY_DECAY = 0.7          # 已訪問區域好奇心衰減率

# VLM參數
VLM_MODEL = "llama3.2-vision"
OLLAMA_ENDPOINT = "http://localhost:11434"
VLM_TEMPERATURE = 0.3          # 降低溫度增加決策確定性

# 地圖參數
JACCARD_THRESHOLD = 0.5        # 區域聚類相似度閾值
ZONE_CLUSTER_MIN_SIZE = 2      # 最小聚類大小
```

---

## 📊 已知問題與改進計畫

### 優先級 1：物體檢測錯誤 (最嚴重)
- ✅ **TASK 1** — 提高 & 自適應閾值 (快速改進)
- ⏳ **TASK 2** — 返回更多檢測結果 (TOP_K 增加至 10-15)
- ⏳ **TASK 3** — 目標分解改進 (更多類別變體)
- ⏳ **TASK 4** — VLM 驗證檢測 (false positive 過濾)
- ⏳ **TASK 5** — OCR 驗證迴圈 (讀取簽牌確認 / 排除)

### 優先級 2：VLM 決策幻覺
- ⏳ **TASK 6** — 子任務分解 (PlanVLM)
- ⏳ **TASK 7** — 多步 correction 機制

### 優先級 3：低效探索
- ⏳ **TASK 8** — 好奇心評分 (Curiosity Value Map)
- ⏳ **TASK 9** — 區域預測評分 (PredictVLM)

---

## 🔧 故障排除

### 問題 1：Ollama 連線失敗
```
Error: Failed to connect to Ollama at http://localhost:11434
```

**解決方案**：
```bash
# 確保 Ollama 服務運行
ollama serve

# 檢查連線
curl http://localhost:11434/api/tags
```

### 問題 2：GroundingDINO 檢測結果為空
```
INFO: No objects detected in photo. Retrying with lower threshold...
```

**解決方案**：
1. 檢查 `server/config.py` 中的 `FALLBACK_THRESHOLD`
2. 確認目標分解是否生成正確的類別名稱
3. 驗證照片質量 (亮度、清晰度)

### 問題 3：OCR 讀取不清楚
**解決方案**：
- 提高照片解析度
- 確保走道簽在照片中佔 > 5% 的面積
- 檢查照片角度 (避免極端仰角或俯角)

---

## 📈 性能指標

| 指標 | 目標 | 目前 |
|------|------|------|
| 平均定位時間 | < 5 秒 | 4.2 秒 |
| 檢測精度 (mAP@0.5) | > 85% | 78% |
| 導航成功率 | > 90% | 72% |
| 客戶滿意度 | > 4.5/5 | 3.8/5 |

---

## 📝 常見命令

```bash
# 測試服務器健康狀態
curl http://localhost:8000/health

# 查看可用的店鋪地圖
curl http://localhost:8000/stores

# 列出會話歷史
curl http://localhost:8000/sessions?store_id=my_store

# 下載標註照片
curl http://localhost:8000/results/annotated_sess_abc123.jpg -o result.jpg
```

---

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

### 開發環境設置
```bash
pip install -r requirements-dev.txt
pytest tests/
```

---

## 📄 授權

MIT License - 詳見 LICENSE 文件

---

## 📞 支持

如有問題，請：
1. 查看 [常見問題](./docs/FAQ.md)
2. 提交 [Issue](https://github.com/B1222017/NAVIGATION-GroundingDINO-/issues)
3. 聯絡開發團隊
