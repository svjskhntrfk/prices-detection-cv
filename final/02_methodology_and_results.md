# Методология, результаты и ручная проверка

## 1. Методология
- Подход: гипотезно-модульная оптимизация OCR-пайплайна.
- Базовая единица сравнения: `run_bundle`/`run_bundle_campaign` с фиксированным `sample-size=96`.
- Критерий выбора (без GT): `case_proxy_v2`.
- Дополнительные контрольные метрики:
  - `proxy_score`
  - fill-rate для `product_name`, `price_any`, `barcode`
  - runtime (`elapsed_sec`)

### Champion-конфигурация
`data_input/H2, preprocess/H1, ocr/H4, parsers/H2, parsers/H4, qr_barcode/H1, track_merge/H1`

## 2. Ограничения
- Размеченный GT в текущем пакете отсутствует, поэтому финальный выбор не по official `final_score`, а по `case_proxy_v2`.
- Результат зафиксирован на `sample=96` (лучший завершённый run на момент упаковки).

## 3. Результат
- Champion-run: `bundle_omega2_sample_20260518_214211`
- Основные значения:
  - `case_proxy_v2=0.55219`
  - `proxy_score=0.66510`
  - `product_name fill=0.69792`
  - `price_any fill=0.85417`
  - `barcode fill=0.26042`

## 4. Как проверить результат руками

### 4.1 Проверка артефактов
1. Убедиться, что в `final` есть файлы `01..09`.
2. Открыть `04_result_best.csv` и проверить, что CSV читается (96 строк данных + header).
3. Проверить `05_metrics_best.json`:
   - `rows = 96`
   - `case_proxy_v2 = 0.55219` (округлённо)
   - `proxy_score = 0.66510` (округлённо)

### 4.2 Проверка качества на изображениях
1. В `04_result_best.csv` выбрать 10–15 случайных `filename` (`track_XXXXX`).
2. Для каждого `track_XXXXX` открыть соответствующие изображения из `top_crops/track_XXXXX/`.
3. Сверить руками:
   - `product_name`
   - цены (`price_default`, `price_discount`, `price_card`)
   - `barcode`
4. Подтвердить, что в большинстве кейсов критичные поля заполнены корректно.

### 4.3 Проверка воспроизводимости
1. Открыть `03_ultimate_pipeline.ipynb`.
2. В Kaggle выполнить bootstrap-ячейку зависимостей и затем `Run all`.
3. Сравнить метрики с `05_metrics_best.json` (допустим небольшой дрейф, но значения должны быть близкими).
