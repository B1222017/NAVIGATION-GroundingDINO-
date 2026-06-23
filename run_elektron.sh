#!/bin/bash
# Navigate 24 photos — goal: 電腦
# Run interactively: bash run_elektron.sh
# Or from Claude Code prompt: ! bash run_elektron.sh

set -e
cd "$(dirname "$0")"

PHOTO_DIR="/Users/shingchou/Downloads/Navigation-main-3/photo/系辦"
GOAL="電腦"

run_step() {
  local photo="$1"
  local text="$2"
  echo ""
  echo "════════════════════════════════════════"
  echo "  📷  $photo"
  echo "════════════════════════════════════════"
  python3 navigate_one_by_one.py \
    --goal "$GOAL" \
    --photo "$PHOTO_DIR/${photo}.jpg" \
    --text "$text" 2>&1 \
    | grep -Ev "^(hugging|To disable|Avoid|\t- Avoid|\t- Explicit|tokenizers:)"
  # 若已宣告抵達就停止
  if python3 -c "import json; s=json.load(open('session_navigate.json')); exit(0 if s.get('arrived') else 1)" 2>/dev/null; then
    echo ""
    echo "🏁  已找到目標，導航結束。"
    exit 0
  fi
}

run_step "IMG_1433" "剛進大廳，我該往哪個方向走才能找到電腦？"
run_step "IMG_1435" "沿著接待區旁邊走，我應該往走廊深處去嗎？"
run_step "IMG_1438" "走廊兩側都是辦公室，我應該繼續直走嗎？"
run_step "IMG_1441" "前方還是走廊，有沒有走錯方向？"
run_step "IMG_1498" "看到很多教師辦公室，電腦可能在這附近嗎？"
run_step "IMG_1500" "繼續往前，這條路有什麼新的標示？"
run_step "IMG_1502" "看到會議室了，要繼續直走還是轉彎？"
run_step "IMG_1503" "走廊還在延伸，左右有沒有出口？"
run_step "IMG_1506" "這裡有其他岔路嗎，還是繼續直走？"
run_step "IMG_1510" "這個位置有書架跟辦公桌，有看到電腦在哪嗎？"
run_step "IMG_1511" "繼續往前走，這個方向感覺對嗎？"
run_step "IMG_1513" "看到一排看起來像儲物區的地方，電腦會在這嗎？"
run_step "IMG_1471" "走廊右邊有木製儲物櫃，我該怎麼繼續走？"
run_step "IMG_1468" "一整排儲物櫃，電腦不太可能在這裡，要折返嗎？"
run_step "IMG_1467" "這裡只有辦公室的門，要繼續走嗎？"
run_step "IMG_1516" "看到會客室的標示，方向是這邊嗎？"
run_step "IMG_1517" "這裡有教授辦公室門牌，電腦區通常在哪個方向？"
run_step "IMG_1518" "繼續沿走廊走，有沒有看到電腦設備？"
run_step "IMG_1521" "走廊感覺好長，有沒有任何指向電腦的線索？"
run_step "IMG_1522" "牆上有海報，有沒有看到任何指示牌？"
run_step "IMG_1523" "好像回到大廳區域了，有電腦設備嗎？"
run_step "IMG_1524" "這邊有接待桌，電腦設備在服務台附近嗎？"
run_step "IMG_1526" "還在走廊，我是不是走錯路了？"
run_step "IMG_1528" "前面看起來有設備區，這裡是電腦所在嗎？"

echo ""
echo "⚠  走完全部 24 張照片，導航尚未宣告完成。"
