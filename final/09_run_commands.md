# Команды запуска и проверки

## 1) Один ноутбук (рекомендуется)
Есть единый notebook-блок OCR в:
- `final.ipynb` (в конце файла, секция `OCR CHAMPION SECTION`)
- `final/03_ultimate_pipeline.ipynb` (та же секция как standalone)

Запуск локально:
1. Откройте `final.ipynb`.
2. Прокрутите до секции `OCR CHAMPION SECTION`.
3. Запустите ячейки секции сверху вниз.

Запуск в Kaggle:
1. `git clone` проект в `/kaggle/working/LentaHack26` или `/kaggle/working/LentaHack26b`.
2. Откройте `final.ipynb` (или `final/03_ultimate_pipeline.ipynb`).
3. Запустите первую ячейку секции (bootstrap зависимостей).
4. Сделайте `Restart session`, затем `Run all` по секции.
5. Если датасет не прикреплён через UI Kaggle, задайте переменную:

```python
import os
os.environ["KAGGLE_DATASET_SLUG"] = "owner/dataset-name"
```

## 2) Опциональный CLI запуск (если runner-файлы присутствуют)
Если в проекте есть `run_champion_pipeline.py` или `hypothesis_campaign.py`, notebook сам вызовет их.
Отдельный ручной запуск обычно не нужен.

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
