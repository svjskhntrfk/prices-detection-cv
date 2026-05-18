# Команды запуска и проверки

## 1) Kaggle: быстрый запуск через notebook
1. Загрузить проект в `/kaggle/working/LentaHack26`.
2. Открыть `final/03_ultimate_pipeline.ipynb`.
3. Выполнить первую bootstrap-ячейку (она докачает зависимости).
4. После установки сделать `Restart session` и выполнить `Run all`.

## 2) Локальный запуск champion-конфигурации
Из корня проекта:

```bash
python3 hypothesis_campaign.py run_bundle \
  --project-root . \
  --dataset-root ./top_crops \
  --task-path ./lenta_tech_life_hack_text.md \
  --notebook notebookc9d692d630.ipynb \
  --output-root ./remote_outputs/final_repro_omega2 \
  --mode sample \
  --sample-size 96 \
  --visual-panel-size 24 \
  --seed 123 \
  --timeout -1 \
  --jupyter-cmd jupyter \
  --products-dict-csv ./products_v2_merged.csv \
  --google-dict-csv ./google_dict_normalized.csv \
  --from-scratch \
  --guardrail-tolerance-pp 1.0 \
  --bundle "omega2_fixed=data_input/H2,preprocess/H1,ocr/H4,parsers/H2,parsers/H4,qr_barcode/H1,track_merge/H1"
```

## 3) Быстрая проверка результата и схемы

```bash
python3 - <<'CHECK_EOF'
import csv
import json
from pathlib import Path

csv_path = Path('final/04_result_best.csv')
metrics_path = Path('final/05_metrics_best.json')

assert csv_path.exists(), csv_path
assert metrics_path.exists(), metrics_path

with csv_path.open(newline='', encoding='utf-8-sig') as f:
    rows = list(csv.reader(f))
header = rows[0]
row_count = len(rows) - 1

metrics = json.loads(metrics_path.read_text())

print('rows_csv =', row_count)
print('rows_metrics =', metrics.get('rows'))
print('case_proxy_v2 =', metrics.get('case_proxy_v2'))
print('proxy_score =', metrics.get('proxy_score'))
print('columns_count =', len(header))
CHECK_EOF
```
