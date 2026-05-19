# Финальный отчёт по OCR-пайплайну (LentaHack26)

## 1. Что сделано
- Проанализирован и расширен исходный OCR-пайплайн в `notebookc9d692d630.ipynb` через оркестратор `hypothesis_campaign.py`.
- Добавлены и протестированы продвинутые гипотезы по блокам OCR, парсеров, штрихкодов, коррекции названий и track merge.
- Проведены пакетные эксперименты на двух серверах:
  - `81.26.183.234` (secondary)
  - `81.26.189.209` (primary)
- Для релиза выбран лучший **завершённый** run на момент фиксации.

## 2. Выбранный champion-run
- Run: `bundle_omega2_sample_20260518_214211`
- Путь: `remote_outputs/experiments_ultimate_secondary_v4/bundle_omega2_sample_20260518_214211`
- Метрики:
  - `rows=96`
  - `proxy_score=0.66510`
  - `case_proxy_v2=0.55219`
  - `product_name fill=0.69792`
  - `price_any fill=0.85417`
  - `barcode fill=0.26042`

## 3. Сравнение ключевых прогонов

| Сервер | Run | rows | proxy_score | case_proxy_v2 | product_name | price_any | barcode | elapsed_sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| secondary | baseline | 96 | 0.54167 | 0.45189 | 0.62500 | 0.62500 | 0.20833 | 724.12 |
| secondary | omega1 | 96 | 0.66510 | 0.55203 | 0.69792 | 0.85417 | 0.26042 | 2181.69 |
| secondary | **omega2 (champion)** | 96 | **0.66510** | **0.55219** | **0.69792** | **0.85417** | **0.26042** | 1794.26 |
| primary | baseline | 96 | 0.51406 | 0.43554 | 0.59375 | 0.60417 | 0.17708 | 642.85 |
| primary | ultra1 | 96 | 0.64948 | 0.53950 | 0.66667 | 0.88542 | 0.19792 | 1733.15 |
| primary | ultra2 | 96 | 0.64427 | 0.52871 | 0.64583 | 0.88542 | 0.21875 | 1933.34 |

## 4. Почему выбран именно omega2
- Максимальный `case_proxy_v2` среди завершённых прогонов.
- Такая же сильная заполняемость ключевых полей, как у omega1, но при меньшем runtime.
- Стабильный результат без деградации в нулевые метрики.

## 5. Что положено в папку `final`
- `01_report.md` — этот отчёт.
- `02_methodology_and_results.md` — методология + как проверять руками.
- `03_ultimate_pipeline.ipynb` — воспроизводимый notebook с Kaggle bootstrap.
- `04_result_best.csv` — итог champion-run.
- `05_metrics_best.json` — метрики champion-run.
- `06_run_meta_best.json` — метаданные champion-run.
- `07_requirements_kaggle.txt` — зависимости для Kaggle.
- `08_requirements_local.txt` — зависимости для локального запуска.
- `09_run_commands.md` — команды запуска и проверки.

## 6. Статус фоновых процессов
- Долгие campaign/wave2 процессы остановлены перед финализацией, чтобы зафиксировать релизный срез и не смешивать артефакты.
