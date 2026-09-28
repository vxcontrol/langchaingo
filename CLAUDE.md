# Память проекта

## Текущая задача: ревью релиза PAGI-125 (PR #1)

Состояние на 2026-09-28. Когда ревью закончено, этот раздел можно удалить.

### Что ревьюим

- PR: https://github.com/vxcontrol/langchaingo/pull/1, ветка `fix/PAGI-125-release-review-findings` → `main-vxcontrol`.
- База — предыдущий релиз: `origin/main-vxcontrol` @ `da7016e`. Это ровно merge-base, и ветка от базы не отстаёт,
  поэтому `git diff origin/main-vxcontrol...origin/fix/PAGI-125-release-review-findings` совпадает с diff'ом
  сквош-коммита и с вкладкой «Files changed» в PR. Сквош-ветка не нужна.
- По коммитам не ревьюим: 816 коммитов (198 из них merge), многие «fix»-коммиты правят предыдущие коммиты той же ветки.
- Отревьюенная голова PR: `355dc71`. Если голова сдвинулась, новые изменения смотреть отдельно: `git diff 355dc71..<голова>`.

Объём diff'а:

| Что | Файлов | Строк |
|---|---|---|
| Go-код (не тесты) | 130 | +9 262 / −2 863 |
| Go-тесты | 306 | +31 399 / −3 298 |
| Документация и прочее | 27 | ~+180 |
| Кассеты `testdata/*.httprr` | 225 | +214k (~7,7 МБ) — модели не отдавать, проверять скриптами |

Больше всего кода изменилось в `llms/reasoning`, `llms/bedrock`, `llms/openai`, `llms/anthropic`, `llms/googleai`,
`vectorstores/pgvector`.

### Что уже сделано

1. `/code-review max --comment` по PR #1 → [ревью с 15 inline-комментариями](https://github.com/vxcontrol/langchaingo/pull/1#pullrequestreview-5334862871),
   статус COMMENTED, от аккаунта `sirozha`.
   - Прошло одним агентом: без параллельных ревьюеров и без отдельной перепроверки замечаний.
   - Покрыт только код без тестов (~19k строк diff'а). Тесты не ревьюились.
   - Стоимость ≈ $18 по прайсу API (вся сессия на момент записи ≈ $19).
2. На `355dc71`: `go build ./...` и `go vet` проходят. `go test ./llms/...` проходит, кроме тестов Bedrock,
   которые падали из-за `AWS_CA_BUNDLE` в песочнице (не перепроверено).
3. Кассеты (219 добавленных/изменённых файлов): все 110 значений `Authorization` и `X-Goog-Api-Key` — заглушка
   `test-api-key`. Реальных ключей по шаблонам (`sk-`, `AKIA`, `AIza`, `gsk_`, `hf_`, `xai-`, `nvapi-`, `?key=`) нет.

### Замечания из ревью

Строки — на `355dc71`. «проверено» — ревьюер подтвердил временным тестом; ⚠️ — зависит от поведения вендора,
не проверено.

| # | Где | Суть | Статус |
|---|---|---|---|
| 1 | `llms/ollama/ollamallm.go:350` | `think` уходит для любой модели при включённом reasoning; Ollama отвечает 400 для моделей без thinking (`llama3.2`), раньше вызов работал | проверено |
| 2 | `llms/openai/structured_output.go:235` | `TakesNoJSONSchema` смотрит только на имя модели: deepseek/glm отклоняются и на OpenRouter/vLLM, где json_schema есть | проверено |
| 3 | `vectorstores/pgvector/pgvector.go:490` | ключ фильтра, не являющийся идентификатором (`doc-id`), даёт `ErrInvalidFilterKey`; раньше работал | — |
| 4 | `internal/toolcall/arguments.go:30` | аргументы `null` отклоняются; вызов Gemini без аргументов ломает повторную отправку хода (Bedrock так же) | ⚠️ |
| 5 | `llms/openai/openaillm.go:311` | правила DashScope для Qwen срабатывают на `qwen3-<N>b` на любом хосте (LM Studio, vLLM) | проверено |
| 6 | `llms/reasoning/effort_wire.go:12` | `RejectsPenalties` вырезает penalty-параметры для deepseek*/grok* на любом хосте | проверено |
| 7 | `llms/googleai/option.go:52` | лимит вывода по умолчанию 2048 → 16384 и отправляется всегда; Vertex отвечает 400 для моделей с лимитом 8k | ⚠️ |
| 8 | `llms/googleai/googleai.go:1096` | penalty-параметры уходят во все модели Gemini → 400 «Penalty is not enabled» | ⚠️ |
| 9 | `llms/reasoning/claude_capability.go:285` | в `noPrefillClaude` нет алиасов `claude-*-latest` | проверено |
| 10 | `llms/reasoning_support.go:115` | ветка effort caps для OpenAI не ограничена `ProviderOpenAI`; подсказка обещает Bedrock уровни, которые он не шлёт | проверено |
| 11 | `llms/anthropic/anthropicllm.go:922` | tool-сообщение с дополнительной `TextContent` теперь падает | — |
| 12 | `vectorstores/pgvector/metadata_index.go:191` | `ANALYZE` при каждом старте store под advisory lock (производительность) | — |
| 13 | `llms/ollama/ollamallm.go:358` | своя логика gpt-oss дублирует `reasoning.OllamaEffortsFor`/`GptOssEffort`, копии расходятся | — |
| 14 | `llms/options.go:130` | doc-комментарий `GetEffort` оказался над `HasExplicitTokens` | — |
| 15 | `embeddings/jina/options.go:86` | комментарий говорит о размерностях, а код использует эти числа как `BatchSize` | — |

Общий корень #2, #5, #6: правила по имени модели без учёта хоста. Образец host-aware проверки —
`ServedByDeepSeek(model, host)` и `DashScopeRoute(model, host)`.

### Что осталось сделать

1. Независимо перепроверить 15 замечаний, в первую очередь ⚠️ #4, #7, #8 (документация вендоров, живые вызовы,
   если есть ключи). Итог по каждому: подтверждено или ложное срабатывание.
2. Отревьюить тесты (+31k строк, 306 файлов): проверяют ли то, что заявлено; нет ли `t.Skip`, прячущих падения;
   соответствуют ли тесты своим кассетам.
3. Скриптом, без модели: у каждой кассеты есть тест, нет осиротевших кассет и тестов без кассет.
4. Перезапустить тесты Bedrock без `AWS_CA_BUNDLE` и убедиться, что падение было только из-за песочницы.
5. По итогам пунктов 1–2 ответить в тредах ревью PR #1 (ложные — закрыть с пояснением), новые находки — отдельным
   ревью. Перед любой публикацией на GitHub — подтверждение пользователя.
6. Исправления вносит автор PR в `fix/PAGI-125-release-review-findings`; пушить туда — только с явного разрешения.

### Как запускать

- `/code-review` выполняется форком без Agent tool, то есть одним агентом. Для параллельного ревью нужен
  Workflow; многоагентный запуск — только с явного согласия пользователя в текущей сессии.
- Разбиение для параллельного ревью (~8 агентов + 1–2 на перепроверку): reasoning · openai · bedrock · anthropic ·
  googleai + mistral + huggingface · ядро llms (`options`, `turn`, `warning`, `errors`, `structuredoutput`,
  `internal/toolcall`, ollama) · pgvector + chains + embeddings · тестовая инфраструктура (`testing/llmtest`,
  `internal/httprr`, `internal/devtools`).
- Оценки по прайсу API: перепроверка замечаний + ревью тестов ≈ $15–40; полное параллельное ревью кода и тестов
  с перепроверкой ≈ $40–80; ревью по коммитам ≈ $300–900 (не делать).
- 28.09 недельный лимит аккаунта был в статусе предупреждения, сброс 2026-10-04 00:00 UTC.
