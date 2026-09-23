# OCR-прогон корпуса RAG

Дата: 2026-09-23. Это продолжение [аудита корпуса](rag-corpus-stage-6-audit-2026-09-23.md): OCR применён к страницам PDF без извлекаемого текста, включая страницы внутри смешанных PDF. Исходные файлы корпуса не изменялись.

## Окружение и способ прогона

- Локально установлены Tesseract 5.5.3 с языками `rus`, `eng` и `pdftoppm` 26.08.0. Для macOS установка: `brew install tesseract tesseract-lang poppler`.
- В Dockerfile уже были `tesseract-ocr`, `tesseract-ocr-rus`, `tesseract-ocr-eng`; добавлен `poppler-utils`, необходимый для рендера страниц PDF перед OCR.
- OCR выбирает только страницы, для которых `pypdf.extract_text()` не дал текста. Страницы с исходным текстовым слоем сохраняются в исходном порядке. На PDF заданы лимиты: 30 OCR-кандидатов и 120 секунд. По умолчанию OCR остаётся выключенным и включается флагом `--enable-ocr`.
- Отпечаток 970 поддерживаемых файлов до и после прогона совпал: `e508f3f651b156da1ae47fee75187c48065b886e733d668a56b4e0abe7a1ecf0` (алгоритм указан в исходном аудите).

Команды:

```bash
DOCUMENT_OCR_MAX_PAGES=30 DOCUMENT_OCR_TIMEOUT_SECONDS=120 PYTHONPATH=. \
  uv run --no-sync python -m src.server.app.index_documents \
  --input-dir data_and_documents --audit-only --enable-ocr \
  --report-json .omx/reports/rag-corpus-stage-6-ocr-audit-2026-09-23.json

DOCUMENT_OCR_MAX_PAGES=30 DOCUMENT_OCR_TIMEOUT_SECONDS=120 PYTHONPATH=. \
  uv run --no-sync python -m src.server.app.index_documents \
  --input-dir data_and_documents --persist-dir src/server/chrome_langchain_db \
  --enable-ocr \
  --report-json .omx/reports/rag-corpus-stage-6-ocr-index-2026-09-23.json
```

Перед записью сделана копия прежнего рабочего индекса в `.omx/backups/chrome-before-ocr-2026-09-23`. Прогон выполнен без `--rebuild` и завершился с кодом 0.

## Результат

| Показатель | До OCR | После OCR | Изменение |
| --- | ---: | ---: | ---: |
| Успешно разобранных и проиндексированных документов | 802 | 881 | +79 |
| Чанков в FTS и Chroma | 20 353 | 21 127 | +774 |
| Неразобранных файлов | 168 | 89 | −79 |
| PDF без текстового слоя, распознанных для индекса | 0 из 79 | 79 из 79 | +79 |

По итоговому JSON-отчёту OCR потребовался 84 PDF: 79 полностью без текстового слоя и 5 смешанным. Из 178 страниц-кандидатов 176 дали текст, 2 остались пустыми; пропущенных страниц не было. OCR извлёк 290 636 символов. Две пустые страницы находятся в `from_parsers/studyspravka_distance_stud_proctor.pdf` (страница 4) и `from_parsers/studyspravka_perbud_spb.pdf` (страница 10); обычный текст этих смешанных PDF сохранён. Оставшиеся 89 отказов — пустые DOCX, перечисленные в исходном аудите.

Поле отчёта `no_extractable_text_rate=0.1732` сохраняет долю исходных файлов без текстового слоя до OCR; после восстановления PDF фактическая доля неразобранных файлов равна 89/970 = 9,18%.

Контрольное чтение баз после завершения: `lexical_documents=881`, `lexical_chunks=21127`, таблица Chroma `embeddings=21127`. Ранее пустой `from_parsers/11.Выписка УС 29.11.23 № 14-О внес.изм.в Положение об акад.мобил.pdf` присутствует в обоих индексах с 6 чанками; первый начинается с «Национальный исследовательский университет». В смешанном `from_parsers/1130608174.pdf` распознанная страница 1 есть и в FTS, и в Chroma с меткой `page=1`.

## Проверки и границы результата

- `PYTHONPATH=. uv run --no-sync pytest -q` с отдельными временными `LEXICAL_INDEX_PATH` и `WEB_AUTH_DATABASE_URL`: **233 passed**.
- `uv run --no-sync ruff check`, `black --check`, `ty check` для изменённого Python-кода: без замечаний. `git diff --check` для затронутых отслеживаемых файлов: без замечаний.
- `docker build --target runtime-base` завершился успешно. В проверочном контейнере доступны Tesseract 5.5.0, языки `eng` и `rus`, `pdftoppm` 25.03.0. Временный тестовый тег образа удалён; полный серверный образ не собирался.
- Независимое ревью подтвердило, что сбой Tesseract после успешного рендера страницы больше не запускает резервную ветку `page.images`; регрессионный тест покрывает этот случай.
- Итоговые JSON: `.omx/reports/rag-corpus-stage-6-ocr-audit-2026-09-23.json` и `.omx/reports/rag-corpus-stage-6-ocr-index-2026-09-23.json`. Журнал полной индексации: `.omx/logs/rag-corpus-stage-6-ocr-index-2026-09-23.log`.

OCR увеличил объём индексируемого текста, но возможны ошибки распознавания символов; он не подтверждает актуальность документов и не доказывает улучшение top-5 или качества ответов. Эти свойства нужно измерять отдельно на зафиксированном корпусе.
