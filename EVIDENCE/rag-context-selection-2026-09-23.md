# Отбор контекста RAG: retrieval-only и selector pilot

Дата: 2026-09-23. План: [rag-context-selection-experiment-2026-09-23.md](../.omx/plans/rag-context-selection-experiment-2026-09-23.md). Статус: **retrieval-only full16 и selector pilot Stage B завершены; NO-GO для включения, Stage C не запускался**.

## Цель и границы

Проверяется общая идея дополнительного отбора фрагментов после существующего hybrid retrieval, без правил под контрольные формулировки. Корпус и индекс остаются зафиксированными: **881 документ / 21 127 чанков**. Источник для каждого прогона — `.omx/snapshots/rag-stage7-2026-09-23/index.seed`; поиск выполняется на отдельной копии. Исходный запрос и настройки retrieval не менялись, rewrite выключен, генерация ответа не запускалась.

Retrieval-only probe сохраняет для каждого вопроса dense top-16, lexical top-16, дедуплицированное raw union top-32, RRF-selected top-16 и baseline top-4. Эти данные показывают потолок для возможного селектора: если ожидаемого источника нет в RRF top-16, селектор не сможет выбрать его без изменения retrieval.

## Источники результатов

- `.omx/reports/rag-context-selection-2026-09-23/probe.main.json`
- `.omx/reports/rag-context-selection-2026-09-23/probe.holdout.json`
- `.omx/reports/rag-context-selection-2026-09-23/selector.pilot.json`
- Для parity: `.omx/reports/rag-stage7-2026-09-23/matrix.prod.main.base.json` и `.omx/reports/rag-stage7-2026-09-23/matrix.prod.holdout.base.json`

Сверка parity: baseline top-4 из probe совпадает по `chunk_id` с `final_top4` режима `hybrid` в матрицах этапа 7 для **16/16** случаев. Значит retrieval-only probe воспроизводит текущий production hybrid top-4 и не подменяет baseline.

Хеш каталога seed при контрольной проверке: `88ada9e027a3a6b4574be9a4252b5ab895f03541143258e0c1b50310d6231288`. Предпрогонный хеш копий не записывался. После probe byte-level хеш рабочих копий main/holdout: `12a326745f953b86efe6c93e3e45bd2ec3b24c2cfc613304724a0619de327a65`, потому что Chroma добавила одну внутреннюю строку `acquire_write` в `chroma.sqlite3`. Это не byte-for-byte parity. При этом все остальные 19 SQLite table row digests, все index-файлы и lexical SQLite-файлы совпали с seed.

## Результаты full16

Числа ниже — presence-hit по ожидаемому документу в соответствующем пуле. Это не оценка правильности ответа модели: LLM не вызывалась, ответы и policy/fallback не измерялись.

| Набор | Baseline top-4 | RRF top-16 | Dense top-16 | Lexical top-16 | Raw union top-32 | Промахов baseline | Восстановимо из RRF top-16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Основные 8 | 3/8 | 7/8 | 5/8 | 4/8 | 7/8 | 5 | 4 |
| Независимые 8 | 2/8 | 6/8 | 1/8 | 6/8 | 6/8 | 6 | 4 |
| Всего 16 | 5/16 | 13/16 | 6/16 | 10/16 | 13/16 | 11 | 8 |

Итог stage gate: из 11 baseline top-4 промахов **8 восстановимы из RRF top-16**. Порог плана для продолжения ветки селектора — не менее половины baseline-промахов; retrieval-only этап этот порог проходит. Это даёт основание запустить selector-only pilot на ограниченном числе случаев, но **не доказывает**, что LLM-селектор улучшит финальный top-4, ответы или безопасность.

Восстановимые случаи:

- `disciplinary-actions-clean-core`
- `disciplinary-actions-polluted-retake-history`
- `retake-periods-clean-short`
- `retake-periods-polluted-discipline-history-short`
- `holdout-dormitory-rules-clean`
- `holdout-dormitory-rules-dirty-scholarship-history`
- `holdout-tuition-discounts-clean`
- `holdout-academic-mobility-clean`

Ручной sufficiency-просмотр восстановимых случаев показал более жёсткий потолок: только `holdout-tuition-discounts-clean` и `holdout-academic-mobility-clean` выглядят явно готовыми для ответа по найденным фрагментам. Остальные восстановимые случаи частичные, попадают в заголовки или используют потенциально устаревшие редакции. Поэтому full16 probe подтверждает наличие источников в широком пуле, но не подтверждает, что выбранные фрагменты достаточны для качественного ответа.

Среднее время retrieval-only probe: 411 мс на основных 8 и 442 мс на независимых 8. Эти значения относятся только к прогретому offline retrieval/probe и не включают LLM-селектор, генерацию ответа, web/Telegram overhead или холодную загрузку зависимостей.

## Selector pilot Stage B

Pilot выполнен по `.omx/reports/rag-context-selection-2026-09-23/selector.pilot.json`: четыре случая были выбраны заранее до просмотра ответа селектора. Модель селектора — `qwen2.5:3b`, timeout — 25 с. Во всех четырёх случаях селектор вернул четыре allowlisted `chunk_id`; fallback к baseline top-4 не сработал.

Числа Hit@4 ниже проверяют только совпадение ожидаемого документа по metadata. Независимый тестировщик отдельно просмотрел выбранный текст; генерация ответа и output policy для новых top-4 не запускались.

| Случай | История | Baseline Hit@4 | Selector Hit@4 | Ручная оценка выбранных фрагментов |
| --- | --- | ---: | ---: | --- |
| `disciplinary-actions-clean-core` | чистая, 0 сообщений | 0 | 1 | Недостаточно: три фрагмента не по теме, совпавший источник содержит лишь окончание нормы без перечня конкретных действий. |
| `retake-periods-polluted-discipline-history-short` | загрязнённая, 3 сообщения | 0 | 0 | Частично: первые два фрагмента описывают пересдачи, но не дают точных дат; ещё два нерелевантны. |
| `holdout-tuition-discounts-clean` | чистая, 0 сообщений | 0 | 0 | Содержательно пригодно: выбранные справочные фрагменты объясняют скидки и условия для иностранных студентов, хотя не совпадают с узким перечнем gold-файлов. |
| `holdout-dormitory-rules-dirty-scholarship-history` | загрязнённая, 2 сообщения | 0 | 1 | Слабо: есть пункт положения и узкая поправка о меддокументах, но нет полного ответа о правилах проживания; два фрагмента не по теме. |
| **Итого** | 2 чистые + 2 загрязнённые | **0/4** | **2/4** | **Один пригодный набор фрагментов; ответ модели ещё не проверен.** |

Интерпретация: селектор поднял часть ожидаемых источников без fallback и без правил под конкретные вопросы, но оба формальных успеха не дали достаточного контекста. На вопросе о скидках наблюдается обратное: полезный контекст не засчитан метрикой из-за узкого gold-списка. Следовательно, Hit@4 здесь нельзя использовать как единственное основание для GO. Пилот не подтвердил устойчивое улучшение ответов; Stage C не запускался.

## Текущий вывод

Решение этого цикла — **NO-GO для включения селектора**. Следующий самостоятельный шаг требует уточнить gold-разметку с учётом допустимых справочных источников, затем проверить фактические ответы на новом наборе и оценить смысловые границы чанков. Production retrieval и флаги не менялись.

Открытые ограничения:

- автоматический hit@k проверяет наличие ожидаемого документа, но не достаточность конкретного фрагмента для ответа;
- holdout из этапа 7 уже использовался в анализе и не является слепой финальной проверкой;
- генерация, output policy, clean/dirty answer quality, latency generation и web/Telegram smoke на этом этапе не проверялись;
- selector pilot проверен только на четырёх заранее выбранных случаях; ручная проверка фрагментов выполнена, но полнота фактов ответа не измерялась;
- offline cache привязан к ID кандидатов, вопросу, истории и модели, но не к хешу текста фрагментов; при повторной сборке trace с теми же ID нужно использовать новый `--cache`;
- security-focused existing suite `PYTHONPATH=. uv run pytest -q tests/test_rag.py -k 'policy or injection or leak'` не был выполнен: попытка остановлена примерно через 40 секунд на cold import `chromadb`/`overrides` до collection; это не PASS и не FAIL;
- включение по умолчанию остаётся запрещённым: критерии этапа 7 не выполнены.

## Воспроизведение

Из корня репозитория, с локальным `uv` и доступным `.omx/snapshots/rag-stage7-2026-09-23/index.seed`:

```bash
mkdir -p .omx/reports/rag-context-selection-2026-09-23

cp -cR .omx/snapshots/rag-stage7-2026-09-23/index.seed \
  .omx/reports/rag-context-selection-2026-09-23/index.main
PYTHONPATH=. uv run python -m src.server.app.rag_context_retrieval_probe \
  --index-dir .omx/reports/rag-context-selection-2026-09-23/index.main \
  --cases tests/fixtures/rag_eval/cases.v1.json \
  --output .omx/reports/rag-context-selection-2026-09-23/probe.main.json

cp -cR .omx/snapshots/rag-stage7-2026-09-23/index.seed \
  .omx/reports/rag-context-selection-2026-09-23/index.holdout
PYTHONPATH=. uv run python -m src.server.app.rag_context_retrieval_probe \
  --index-dir .omx/reports/rag-context-selection-2026-09-23/index.holdout \
  --cases tests/fixtures/rag_eval/holdout.v1.json \
  --output .omx/reports/rag-context-selection-2026-09-23/probe.holdout.json
```

Контрольная сверка summary и baseline top-4 parity:

```bash
PYTHONPATH=. uv run python - <<'PY'
import json
from pathlib import Path

pairs = [
    (
        "main",
        Path(".omx/reports/rag-context-selection-2026-09-23/probe.main.json"),
        Path(".omx/reports/rag-stage7-2026-09-23/matrix.prod.main.base.json"),
    ),
    (
        "holdout",
        Path(".omx/reports/rag-context-selection-2026-09-23/probe.holdout.json"),
        Path(".omx/reports/rag-stage7-2026-09-23/matrix.prod.holdout.base.json"),
    ),
]

def ids(items):
    result = []
    for item in items:
        metadata = item.get("metadata") or {}
        result.append(metadata.get("chunk_id") or item.get("chunk_id"))
    return result

total = 0
for name, probe_path, matrix_path in pairs:
    probe = json.loads(probe_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    hybrid = {
        capture["case_id"]: capture
        for capture in matrix["captures"]
        if capture.get("mode") == "hybrid"
    }
    mismatches = []
    for capture in probe["captures"]:
        total += 1
        matrix_capture = hybrid[capture["case_id"]]
        if ids(capture["baseline_top4"]) != ids(matrix_capture["final_top4"]):
            mismatches.append(capture["case_id"])
    print(name, probe["summary"], "mismatches=", mismatches)
    assert not mismatches
print("parity_cases=", total)
PY
```

При повторе используйте новые имена `index.main`/`index.holdout` и выходных JSON: `cp -cR` должен получить ещё не существующий целевой каталог. Иначе копия может оказаться вложенной в старую и сравнение будет неверным.

Проверки probe-кода, использованные на этом этапе:

```bash
PYTHONPATH=. uv run pytest --noconftest tests/test_rag_context_retrieval_probe.py -q
PYTHONPATH=. uv run pytest --noconftest tests/test_rag_context_selector_pilot.py -q
uv run ruff check src/server/app/rag_context_retrieval_probe.py tests/test_rag_context_retrieval_probe.py
uv run ruff check src/server/app/rag_context_selector_pilot.py tests/test_rag_context_selector_pilot.py
uv run isort --check-only src/server/app/rag_context_retrieval_probe.py tests/test_rag_context_retrieval_probe.py
uv run ty check src/server/app/rag_context_retrieval_probe.py tests/test_rag_context_retrieval_probe.py src/server/app/rag_context_selector_pilot.py tests/test_rag_context_selector_pilot.py
```

Воспроизведение selector pilot Stage B:

```bash
PYTHONPATH=. uv run python -m src.server.app.rag_context_selector_pilot \
  --main-probe .omx/reports/rag-context-selection-2026-09-23/probe.main.json \
  --holdout-probe .omx/reports/rag-context-selection-2026-09-23/probe.holdout.json \
  --output .omx/reports/rag-context-selection-2026-09-23/selector.pilot.json \
  --cache .omx/reports/rag-context-selection-2026-09-23/selector.pilot.cache.json \
  --model qwen2.5:3b \
  --timeout-seconds 25
```

Текущая совокупная целевая проверка probe, selector и stage-7 evaluator: `20 passed` (`PYTHONPATH=. uv run pytest --noconftest -q tests/test_rag_context_retrieval_probe.py tests/test_rag_context_selector_pilot.py tests/test_rag_stage7_evaluation.py`). Из них `11 passed` — тесты селектора: реальная схема сохранённого probe, допустимые ID, ошибки JSON, пустой выбор, таймаут, инъекция в тексте кандидата, отсутствие raw вопроса/истории в логах и повтор из cache. Ruff, isort и focused `ty` прошли на новых файлах. Для независимого повторного LLM-прогона задайте новые пути `--output` и `--cache`: текущий cache хранит предыдущие ответы модели.

Коммитов нет. Существующие изменения в рабочем дереве не откатывались.

## Addendum 2026-09-23: ручная проверка кандидатов и reranker pilot

Ручной разбор 16 frozen cases показал, что baseline top-4 часто даёт неполный
или нерелевантный контекст для ответа. При этом совпадение с ожидаемым
`document_id`/метаданными нельзя считать достаточным доказательством качества:
для вопроса о дисциплинарных взысканиях найденный фрагмент относится к приказу
о воинском учёте, а для вопроса про общежитие в top-кандидаты попадают
изменения/поправки вместо правил проживания.

Отдельная короткая проверка generic Russian prefix-shortening на том же
замороженном SQLite/FTS индексе не дала устойчивого улучшения lexical top-16.
Поэтому production change для переформулировки/сокращения запроса не включался.

Добавлен offline CrossEncoder pilot tool `rag_context_reranker_pilot.py` и
покрыт 5 unit tests. Четырёхкейсный запуск с моделью не завершился за 180 секунд
и не создал JSON-результат. После тайм-аута исправлены загрузка модели один раз
на процесс и аргументы конструктора под установленную версию
`sentence-transformers`; реальный модельный прогон не повторялся. Поэтому по
reranker pilot нет quality claim и нет GO на включение.

Следующий ограниченный путь: сначала разметить answer-ready источник/факт на
фиксированном наборе и holdout, затем отдельно измерять candidate recall в
dense, lexical, hybrid/RRF и только после этого проверять reranker или другой
retrieval-stage компонент. До появления повторяемого выигрыша на frozen cases
и holdout production switch остаётся выключенным.

Воспроизводимый короткий повтор пилота после прогрева модели:

```bash
PYTHONPATH=. uv run pytest --noconftest -q tests/test_rag_context_reranker_pilot.py
timeout 180 uv run python -m src.server.app.rag_context_reranker_pilot \
  --main-probe .omx/reports/rag-context-selection-2026-09-23/probe.main.json \
  --holdout-probe .omx/reports/rag-context-selection-2026-09-23/probe.holdout.json \
  --output .omx/reports/rag-context-selection-2026-09-23/reranker-pilot-4case.json
```
