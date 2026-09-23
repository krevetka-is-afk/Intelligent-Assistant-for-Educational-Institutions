# Блок 10 — финализация: доказательства запуска

Дата проверки: 2026-09-22 (Europe/Moscow).

## Статус

Проверены и устранены code-level блокеры запуска. Состояние рабочего дерева,
финальная фиксация ревизии и CI намеренно находятся вне этого отчёта: блок 9
не входит в текущую проверку.

## Среда

- локальный Python: 3.13.9; `uv 0.12.5`;
- собранный образ: Python 3.12.14;
- Docker 29.6.1, Docker Compose v5.2.0;
- модель по умолчанию: `qwen2.5:3b`;
- конфигурация Compose была проверена с временным значением
  `TELEGRAM_SERVICE_KEY`, без использования действующих секретов.

## Автоматические проверки

После устранения блокеров выполнены:

```text
uv lock --check                                      PASS
uv run ruff check .                                  PASS
uv run black --check .                               PASS (46 files unchanged)
uv run isort --check-only .                          PASS (4 files skipped)
uv run -m pytest -q -rs -p no:cacheprovider          PASS (172 passed)
uv run -m pytest --cov=src --cov-report=term-missing
  --cov-fail-under=74 -q                             PASS (172 passed, 80.57%)
```

`docker compose config -q` проверен с временными значениями и для dev-, и для
production-схемы. Без `TELEGRAM_SERVICE_KEY` (включая пустую строку) обе схемы
теперь ожидаемо завершаются до запуска контейнера. Server и bot дополнительно
отклоняют пустые обязательные значения при прямом запуске Python; это покрыто
14 целевыми тестами конфигурации.

Тестовый набор покрывает изоляцию web/Telegram-memory, отклонение неверного
Telegram service key, прямые и косвенные prompt-injection сценарии, fallback
при недоступной LLM и HTML-форматирование Telegram-ответа. В живом стенде
реальный Telegram Bot API не вызывался: действующий `BOT_TOKEN` не нужен и не
использовался.

## Изолированный Compose smoke-test

Стенд запускался отдельным Compose-проектом `finalization-block10`, с
временными тестовыми ключами и собственным volume. До изменения Dockerfile
строгий профиль `cap_drop: ALL` не мог запустить `/app/bin/python`: каталог
`/app` имел права `0700` владельца `appuser`, а entrypoint выполнялся без
capability обхода ACL. Исправление делает только каталог виртуального
окружения проходимым (`chmod 755 /app`), не меняя владельца, пользователя или
capabilities контейнера.

После пересборки:

- `/live` вернул `200` во время подготовки RAG;
- `/ready` сначала вернул `503 initializing`, затем `200 ready` после индексации
  20 353 фрагментов; Docker healthcheck стал `healthy`;
- `/ask` с корректным временным API key вернул `200` и безопасный RAG fallback
  `llm_unavailable` при недоступной Ollama;
- `/web/bootstrap` создал сессию (`303` и cookie), `/web/ask` прошёл как с одной
  cookie, так и с cookie вместе с API key (`200`);
- последовательность Telegram `A1/B1/A2/B2` для двух ID прошла (`200`), а
  неверный `X-Telegram-Service-Key` вернул `401`;
- прямой запрос системного промпта вернул policy-refusal (`200`,
  `policy_forbidden_control_or_secret_request`);
- после restart сначала восстановился `/live`, затем `/ready`; итоговое
  состояние контейнера — `running`, `health=healthy`.

После усиления контракта секретов сервер был собран заново на Python 3.12 и
запущен в отдельном Compose-проекте с временным непустым service key. Он стал
`healthy`, `/ready` вернул `200` после индексации 20 353 фрагментов; валидный
Telegram key дал `200`, а неверный — `401`.

Косвенная инъекция проверена детерминированными unit-тестами: в реальный
учебный корпус документов не добавлялся искусственный вредоносный фрагмент.

## Сопоставление с майским архивом

Архив `intelligent-assistant-for-educational-institutions-selfhost-deploy`
является старым baseline. Полный откат небезопасен: архив не содержит `/live`,
`/ready`, `/telegram/ask`, раздельного Telegram service key,
readiness/policy/validation и ограничений RAG-контекста. Он также вызывает
общий `/ask` от имени Telegram через API key.

## Оставшиеся не-кодовые условия

1. Для реального развёртывания оператор должен передать единый непустой
   `TELEGRAM_SERVICE_KEY` server и bot вне репозитория. При отсутствии значение
   теперь не маскируется: Compose завершится понятной ошибкой.
2. Документация остаётся несинхронизированной с контрактом: ПЗ описывает bot ->
   `/ask` и отсутствие injection-защиты, ПМИ не описывает trusted Telegram и
   `/telegram/ask`/`/live`/`/ready`, РО ссылается на readiness `/health` и
   несуществующий `POST /index`.
3. CI не оценивался по явному ограничению задачи (пропущен блок 9).
