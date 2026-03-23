# Отчёт по ДЗ-3

## 1. Постановка задачи
Рассматривается production-like сценарий `ticket routing`: входящий текстовый тикет необходимо автоматически отправить в правильную очередь поддержки.

Модель:
- `Qwen/Qwen3-1.7B`

Данные:
- `Tobi-Bueck/customer-support-tickets`

Eval-постановка:
- используются только английские тикеты;
- целевая метка качества — `queue`;
- берутся `top-10` самых частых очередей;
- из каждой очереди выбирается по `30` примеров;
- итоговый eval-набор содержит `300` тикетов;
- обучение не проводится, сравниваются только inference-time конфигурации.

## 2. Baseline и ускорение
Baseline:
- `Qwen/Qwen3-1.7B` через `transformers + mps`

Ускоренная версия:
- `Qwen/Qwen3-1.7B-GGUF` через `llama.cpp + Metal`

Инженерная стратегия:
- сохраняется одна и та же задача;
- сохраняется один и тот же eval pipeline;
- сохраняется один и тот же structured output;
- отключается reasoning;
- меняются runtime и формат весов.

Таким образом, стратегия состоит не из одного трюка, а из комбинации:
- GGUF как более компактного представления весов;
- `llama.cpp` как специализированного runtime;
- constrained structured output для стабильного JSON-ответа.

## 3. Ограничения
В работе используются следующие измеримые ограничения:
- `accuracy drop <= 2.0` п.п. относительно baseline;
- `mean latency <= 1800 ms`;
- `p95 latency <= 2500 ms`;
- `throughput >= 0.7 samples/s`;
- `peak process memory <= 8000 MB`;
- `artifact size <= 4000 MB`.

Все ограничения проверяются автоматически в `limits_check.csv`.

## 4. Baseline summary
Архитектурная сводка baseline:
- архитектура: `Qwen3ForCausalLM`
- `hidden_size = 2048`
- `num_hidden_layers = 28`
- `num_attention_heads = 16`
- `vocab_size = 151936`
- `max_input_tokens = 4096`
- `max_new_tokens = 256`

Основные bottlenecks бейзлайна:
- full-precision веса;
- неоптимальный `transformers.generate()`;
- autoregressive decoding даже для короткого structured output.

## 5. План реализации
Порядок реализации:
1. Зафиксировать baseline на `transformers + mps`.
2. Зафиксировать ускоренную конфигурацию `GGUF + llama.cpp`.
3. Запустить обе конфигурации на одном eval-наборе.
4. Сравнить результаты по качеству, latency, throughput, памяти и размеру.
5. Проверить выполнение ограничений.

Критерии остановки:
- потеря по accuracy больше `2` п.п.;
- отсутствие выигрыша по latency и throughput;
- выход за memory budget или size budget.

## 6. Structured output
Во всех конфигурациях используется один и тот же формат ответа:

```json
{"queue":"..."}
```

Для этого:
- prompt жёстко требует JSON;
- reasoning отключён;
- ответы парсятся через JSON-first parser;
- для `llama.cpp` используется grammar-based constrained output.

## 7. Результаты экспериментов
Итоговые результаты на `300` примерах:

| Runtime | Accuracy | Macro-F1 | Parse success | Mean latency, ms | P95 latency, ms | Throughput, samples/s | Artifact size, MB | Peak RSS, MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `transformers + mps` | 0.19 | 0.1429 | 0.9433 | 389.76 | 479.65 | 2.5657 | 3875.24 | 1407.11 |
| `llama.cpp + GGUF Q8_0` | 0.20 | 0.1537 | 1.0000 | 268.52 | 339.12 | 3.7241 | 1749.44 | 3064.20 |


## 8. Сравнение с baseline
Относительно baseline ускоренная конфигурация показывает:
- `accuracy_drop_pp = -1.0`, то есть ускоренная версия оказалась на `1` п.п. лучше baseline;
- `speedup_vs_baseline = 1.45x`;
- `throughput_gain_vs_baseline = 1.45x`;
- уменьшение размера артефакта с `3875 MB` до `1749 MB`.

Главный trade-off:
- `llama.cpp + GGUF` заметно быстрее и компактнее;
- при этом peak RSS выше: `3064 MB` против `1407 MB` у baseline.

Это важный инженерный результат: GGUF сильно уменьшает размер модели на диске, но не гарантирует минимальный peak process memory в данном локальном runtime.

## 9. Проверка ограничений
По итоговой таблице `limits_check.csv` обе конфигурации проходят заданные ограничения.

Для `llama.cpp + GGUF Q8_0`:
- accuracy budget: выполнен;
- mean latency budget: выполнен;
- p95 latency budget: выполнен;
- throughput budget: выполнен;
- memory budget: выполнен;
- size budget: выполнен.

## 10. Интерпретация результатов
Главный практический вывод такой:
- `llama.cpp + GGUF Q8_0` оказался лучше baseline почти по всем ключевым serving-метрикам;
- ускоренная версия быстрее, компактнее по размеру модели и стабильнее по structured output;
- baseline выигрывает только по peak RSS.

При этом необходимо учитывать абсолютное качество:
- `accuracy` около `0.19–0.20` и `macro-F1` около `0.14–0.15` являются низкими;
- задача zero-shot routing по `10` классам для этой модели в текущей постановке остаётся сложной;
- поэтому результат важен прежде всего как инженерное сравнение runtime, а не как готовое production-решение без дальнейшей адаптации модели.

## 11. Вывод
Ограничения в рамках выбранного сценария достигнуты.

Принятые компромиссы:
- выбран ускоренный runtime с лучшими latency/throughput;
- принят более высокий peak RSS по сравнению с baseline;
- сохранён один и тот же eval pipeline для честного сравнения.
