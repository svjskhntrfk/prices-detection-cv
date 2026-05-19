# Final Merge Pack

Этот каталог содержит финальные артефакты и единый notebook-блок для OCR pipeline.

## Главное
- В `final.ipynb` добавлена секция `OCR CHAMPION SECTION`.
- Эта же секция вынесена в `final/03_ultimate_pipeline.ipynb` как standalone-ноутбук.
- Пути в секции сделаны под `local + Kaggle` и поддерживают структуру:
  - `.../LentaHack26`
  - `.../LentaHack26b`

## Структура
```text
final/
├── README.md
├── 01_report.md
├── 02_methodology_and_results.md
├── 03_ultimate_pipeline.ipynb
├── 04_result_best.csv
├── 05_metrics_best.json
├── 06_run_meta_best.json
├── 07_requirements_kaggle.txt
├── 08_requirements_local.txt
└── 09_run_commands.md
```

## Как запускать
- Сценарии запуска и проверки: `final/09_run_commands.md`.
- Рекомендуемый путь: запускать секцию `OCR CHAMPION SECTION` прямо в `final.ipynb`.
