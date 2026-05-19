# evaluate_matching.py — документация

Скрипт сравнивает результаты распознавания ценников с ground truth CSV и выдаёт метрики качества. Поддерживает два режима: полная оценка (детекция + OCR) и только детекция (нашли ли ценник).

---

## Оглавление

1. [Быстрый старт](#быстрый-старт)
2. [Режимы работы](#режимы-работы)
3. [Входные данные](#входные-данные)
4. [Алгоритм матчинга](#алгоритм-матчинга)
5. [Нормализация значений](#нормализация-значений)
6. [Метрики](#метрики)
7. [Выходные файлы](#выходные-файлы)
8. [Веса полей](#веса-полей)
9. [Аргументы командной строки](#аргументы-командной-строки)

---

## Быстрый старт

```bash
# Установить зависимости
pip install pandas

# Полная оценка (детекция + содержимое полей)
python evaluate_matching.py \
  --pred outputs/result.csv \
  --gt 26_12-20.csv \
  --out outputs/report.json \
  --matches-out outputs/matches.csv

# Только детекция (достаточно bbox + timestamp в pred CSV)
python evaluate_matching.py \
  --pred outputs/result.csv \
  --gt 26_12-20.csv \
  --out outputs/report_det.json \
  --matches-out outputs/matches_det.csv \
  --detection-only
```

---

## Режимы работы

### Полная оценка (по умолчанию)

Оценивает и факт нахождения ценника, и качество распознавания текстовых полей (цены, штрихкоды, название товара и т.д.).

Pred CSV должен содержать те же поля, что и GT CSV (или их подмножество).

**Главные метрики:** `final_score`, `weighted_final_score`

### Только детекция (`--detection-only`)

Оценивает только факт нахождения ценника по bbox и временной метке — без проверки содержимого полей.

Pred CSV достаточно с колонками: `filename`, `frame_timestamp`, `x_min`, `y_min`, `x_max`, `y_max` (плюс опциональный `confidence`).

**Главные метрики:** `recall`, `precision`, `f1`

---

## Входные данные

### Ground Truth CSV (`--gt`)

Полный CSV с разметкой. Обязательные колонки для матчинга:

| Колонка | Тип | Описание |
|---|---|---|
| `filename` | str | Имя видеофайла |
| `frame_timestamp` | float (мс) | Временная метка кадра в миллисекундах |
| `x_min`, `y_min`, `x_max`, `y_max` | float | Координаты bounding box ценника |

Контентные колонки (оцениваются в полном режиме):

| Колонка | Тип нормализации |
|---|---|
| `barcode`, `id_sku`, `code`, `qr_code_barcode`, `action_code_qr` | код (убираются спецсимволы) |
| `price_default`, `price_card`, `price_discount`, `price1_qr`…`price4_qr`, `action_price_qr`, `wholesale_level_1_price`, `wholesale_level_2_price` | float (запятая→точка) |
| `product_name`, `color`, `additional_info`, `discount_amount`, `print_datetime`, `special_symbols`, `wholesale_level_1_coun`, `wholesale_level_2_count` | текст (strip + lower) |

### Pred CSV (`--pred`)

Результат работы пайплайна. Должен содержать те же ключевые колонки (`filename`, `frame_timestamp`, bbox). Контентные колонки — по возможности.

Отсутствующие в pred колонки считаются пустыми (`None`) при оценке.

### Значения «нет данных»

Следующие строки во всех полях трактуются как отсутствующее значение и **не штрафуются** при сравнении:

```
"нет", "none", "null", "nan", ""  (а также NaN от pandas)
```

---

## Алгоритм матчинга

Каждая строка pred CSV сопоставляется с одной строкой GT CSV. Каждая GT-строка может быть использована только один раз.

### Pass 1 — матчинг по штрихкоду (только в полном режиме)

Если у pred-строки есть `barcode` и он точно совпадает (после нормализации) с каким-то GT-штрихкодом — это матч. Такие пары имеют `match_score = 1.0`.

Pred-строки без штрихкода или с несовпадающим переходят в Pass 2.

### Pass 2 — пространственно-временной матчинг

Для каждой незаматченной pred-строки среди оставшихся GT-строк ищется лучший кандидат по трём критериям (все должны выполняться):

1. **`filename`** — должны совпадать (если присутствует в обоих)
2. **`|Δt| ≤ time_tolerance_ms`** — разница временных меток в пределах допуска (по умолчанию 500 мс)
3. **`IoU ≥ iou_threshold`** — пересечение bbox-ов не ниже порога (по умолчанию 0.3)

Среди кандидатов выбирается тот, у кого максимальный `score`:

```
score = IoU - 0.1 × (|Δt| / time_tolerance_ms)
```

Pred-строки, не нашедшие пару, получают `match_type = "unmatched"`.

### IoU (Intersection over Union)

```
IoU = площадь пересечения / площадь объединения
```

Если два bbox не пересекаются — IoU = 0.

---

## Нормализация значений

Перед сравнением каждое значение проходит нормализацию в зависимости от типа поля:

**Все поля:**
- `NaN` / `None` → `None`
- `strip()` + `lower()`
- sentinel-строки → `None`

**Цены** (`price_*`, `*_price`, `action_price_qr`):
- запятая → точка, убираются пробелы
- парсится как `float`
- если не парсится — оставляется как строка

```
"3 789,49"  →  3789.49
"189,00"    →  189.0
```

**Коды** (`barcode`, `id_sku`, `code`, `qr_code_barcode`, `action_code_qr`):
- убираются все символы кроме `[a-zA-Z0-9_]`

```
"350 061-011.7022"  →  "3500610117022"
"ABC-123"           →  "abc123"
```

**Текст** (все остальные поля):
- только `strip()` + `lower()`

---

## Метрики

### Полный режим

#### `final_score` — основная метрика

```
final_score = кол-во GT-ценников, у которых:
                  (a) нашёлся матч в pred  И
                  (b) row_accuracy ≥ 0.8
              ──────────────────────────────────
                        всего GT-ценников
```

Диапазон [0, 1]. Значение 1.0 означает, что каждый ценник найден и ≥ 80% его полей распознаны верно.

#### `weighted_final_score` — взвешенная метрика

Аналогично `final_score`, но вместо `row_accuracy` используется `weighted_row_accuracy`:

```
weighted_row_accuracy = Σ weight(f) для правильных полей f
                        ────────────────────────────────────
                        Σ weight(f) для всех оцениваемых полей f
```

Позволяет сделать так, чтобы ошибка в штрихкоде штрафовала сильнее, чем ошибка в дате.

#### `row_accuracy` — точность по строке

```
row_accuracy = кол-во верных полей / кол-во полей, где GT не None
```

Поля, в которых GT пустой (None), не учитываются. Незаматченные строки → 0.0. Строки, где GT везде пустой → 1.0.

#### `field_accuracy` — точность по полю

Для каждого поля отдельно:

```
field_accuracy[field] = кол-во правильных матчей по этому полю
                        ──────────────────────────────────────
                        кол-во GT-строк, где это поле не None
```

#### Счётчики матчинга

| Метрика | Что означает |
|---|---|
| `total_gt` | Всего ценников в GT |
| `total_pred` | Всего ценников в pred |
| `matched_by_barcode` | Сопоставлено через штрихкод (Pass 1) |
| `matched_by_spatial` | Сопоставлено через IoU + timestamp (Pass 2) |
| `unmatched_result` | Pred-строки без пары (ложные детекции) |
| `unmatched_gt` | GT-строки без пары (пропущенные ценники) |

---

### Режим `--detection-only`

| Метрика | Формула |
|---|---|
| `recall` | `matched_gt / total_gt` |
| `precision` | `matched_gt / total_pred` |
| `f1` | `2 × precision × recall / (precision + recall)` |
| `matched_gt` | GT-ценники, у которых нашёлся pred-матч |
| `unmatched_gt` | Пропущенные ценники |
| `false_positives` | Pred-ценники без GT-пары |

---

## Выходные файлы

### `report.json`

Агрегированные метрики всего прогона.

**Полный режим:**

```json
{
  "total_gt": 42,
  "total_pred": 40,
  "matched_by_barcode": 15,
  "matched_by_spatial": 22,
  "unmatched_result": 3,
  "unmatched_gt": 5,
  "final_score": 0.8095,
  "weighted_final_score": 0.7857,
  "field_weights": { "barcode": 3.0, "price_default": 2.0, ... },
  "row_accuracy_threshold": 0.8,
  "time_tolerance_ms": 500.0,
  "iou_threshold": 0.3,
  "field_accuracy": {
    "price_default": 0.9524,
    "barcode": 0.8667,
    ...
  }
}
```

**Режим `--detection-only`:**

```json
{
  "mode": "detection_only",
  "total_gt": 42,
  "total_pred": 40,
  "matched_gt": 37,
  "unmatched_gt": 5,
  "false_positives": 3,
  "recall": 0.881,
  "precision": 0.925,
  "f1": 0.9025,
  "time_tolerance_ms": 500.0,
  "iou_threshold": 0.3
}
```

---

### `matches.csv`

Одна строка на каждую pred-строку. Позволяет разобраться, какая pred-строка с какой GT совпала и какие поля ошиблись.

**Полный режим — колонки:**

| Колонка | Описание |
|---|---|
| `pred_index` | Индекс строки в pred CSV (0-based) |
| `gt_index` | Индекс строки в GT CSV, или `""` если нет пары |
| `match_type` | `"barcode"` / `"spatial"` / `"unmatched"` |
| `match_score` | Качество матча: barcode → 1.0; spatial → IoU − 0.1×(Δt/tol); unmatched → 0.0 |
| `row_accuracy` | Доля верных полей по этой паре (0.0 если unmatched) |
| `weighted_row_accuracy` | Взвешенная доля верных полей (0.0 если unmatched) |
| `field_errors` | Поля, в которых pred ≠ GT, через `\|`. Пусто если всё верно |

**Режим `--detection-only` — колонки:**

| Колонка | Описание |
|---|---|
| `pred_index` | Индекс строки в pred CSV |
| `gt_index` | Индекс строки в GT CSV, или `""` |
| `match_type` | `"spatial"` / `"unmatched"` |
| `iou_score` | IoU для пространственного матча (0.0 если unmatched) |

---

## Веса полей

Используются в `weighted_row_accuracy` и `weighted_final_score`. Захардкожены в `DEFAULT_FIELD_WEIGHTS`:

| Вес | Поля |
|---|---|
| **3.0** | `barcode` |
| **2.0** | `id_sku`, `code`, `qr_code_barcode`, `action_code_qr`, `price_default`, `price_card` |
| **1.5** | `price_discount`, `action_price_qr` |
| **1.0** | `price1_qr`…`price4_qr`, `wholesale_level_1_price`, `wholesale_level_2_price`, `product_name`, `discount_amount` |
| **0.5** | `wholesale_level_1_coun`, `wholesale_level_2_count`, `print_datetime`, `color` |
| **0.25** | `additional_info`, `special_symbols` |
| **1.0** | все остальные поля (дефолтный вес) |

Чтобы изменить веса — отредактировать словарь `DEFAULT_FIELD_WEIGHTS` в начале файла `evaluate_matching.py`.

---

## Аргументы командной строки

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--pred` | обязательный | Путь к CSV с предсказаниями |
| `--gt` | обязательный | Путь к ground truth CSV |
| `--out` | `report.json` | Путь к выходному JSON с метриками |
| `--matches-out` | `matches.csv` | Путь к выходному CSV с детализацией по парам |
| `--time-tolerance-ms` | `500` | Допустимая разница временных меток в мс (Pass 2) |
| `--iou-threshold` | `0.3` | Минимальный IoU для пространственного матча (Pass 2) |
| `--detection-only` | выключен | Оценивать только детекцию (bbox + timestamp), без содержимого |

### Примеры

```bash
# Строже по IoU, шире по времени
python evaluate_matching.py \
  --pred outputs/result.csv \
  --gt 26_12-20.csv \
  --iou-threshold 0.5 \
  --time-tolerance-ms 1000

# Сохранить отчёт в нестандартное место
python evaluate_matching.py \
  --pred outputs/result.csv \
  --gt 26_12-20.csv \
  --out reports/run_01.json \
  --matches-out reports/matches_01.csv

# Только детекция — минимальный pred CSV (bbox + timestamp)
python evaluate_matching.py \
  --pred detections.csv \
  --gt 26_12-20.csv \
  --out det_report.json \
  --matches-out det_matches.csv \
  --detection-only
```
