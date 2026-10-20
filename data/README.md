# Data

Данные не хранятся в Git-репозитории из-за их размера.

Источник: [официальный KION Dataset](https://github.com/irsafilo/KION_DATASET).

## Загрузка

Из корня проекта выполните:

```bash
mkdir -p data/raw

curl -fL --progress-bar \
  -o data/raw/data_original.zip \
  https://raw.githubusercontent.com/irsafilo/KION_DATASET/main/data_original.zip

unzip data/raw/data_original.zip \
  "data_original/interactions.csv" \
  "data_original/users.csv" \
  "data_original/items.csv" \
  -d data/raw
```

После распаковки ожидается следующая структура:

```text
data/raw/data_original/
├── interactions.csv
├── items.csv
└── users.csv
```

## Проверенные характеристики

- взаимодействия: 5 476 251;
- пользователи: 962 179;
- объекты: 15 706;
- период взаимодействий: с 2021-03-13 по 2021-08-22.

## Примечание

Отдельный архив `interactions.csv.zip` содержит 1 594 786 взаимодействий. В проекте используется полная таблица `interactions.csv` из архива `data_original.zip`.

## Первичный аудит качества

Проверка полной таблицы взаимодействий показала:

- 828 пропусков в watched_pct;
- пропусков в остальных колонках нет;
- повторных пар user_id–item_id нет;
- значений watched_pct вне диапазона 0–100 нет;
- неположительных значений total_dur нет;
- медиана числа взаимодействий на пользователя — 2;
- 95-й перцентиль — 21;
- максимум — 1 341.

На этом этапе строки не удаляются.
Пропуски watched_pct сохраняются как неизвестные значения.

Повторный запуск аудита:

```bash
uv run python -m kion_recs.data.audit