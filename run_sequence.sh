#!/bin/bash
# Run full navigation sequence
PHOTO_DIR="/Users/shingchou/Downloads/Navigation-main-3/photo/系辦"

set -e

run_step() {
  local photo="$1"
  local text="$2"
  echo ""
  echo "════ $photo ════"
  python navigate_one_by_one.py \
    --photo "$PHOTO_DIR/${photo}.jpg" \
    --text "$text" 2>&1 \
    | grep -Ev "^(hugging|To disable|Avoid|\t- Avoid|\t- Explicit|tokenizers:)"
}

check_arrived() {
  echo "$1" | grep -q "🏁"
}

OUT=$(run_step "IMG_1434" "我往前走了幾步，請問有看到門牌或指示嗎"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1437" "繼續沿走廊前進，注意查看兩側門牌"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1492" "我轉了個方向繼續走，請告訴我該往哪走"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1456" "我繼續前進，注意左右兩側辦公室門牌"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1455" "我稍微退後一點，仔細看附近的門牌文字"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1457" "繼續往前，請幫我找吳世琳教授的辦公室"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1458" "我又往前走了一步，有看到門牌嗎"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1459" "繼續移動，注意走廊兩側的標示"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1460" "我往前走，請告訴我現在能看到什麼門牌"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1462" "繼續前進，仔細找吳世琳教授的辦公室"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1463" "我靠近了一些，有看到相關標示嗎"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1464" "繼續查看附近的門牌，有看到吳世琳嗎"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1465" "我往前移動了，請告訴我該往哪走"); echo "$OUT"; check_arrived "$OUT" && exit 0
OUT=$(run_step "IMG_1466" "最後一張，這裡有吳世琳教授的辦公室嗎"); echo "$OUT"; check_arrived "$OUT" && exit 0

echo ""
echo "⚠ 走完所有照片，未找到「吳世琳」教授的辦公室。"
