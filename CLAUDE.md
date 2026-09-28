# Память проекта

## Текущая задача: исправления по ревью релиза PAGI-125 (PR #1)

Состояние на 2026-09-28: оба ревью сделаны и опубликованы, замечания отсортированы, исправления ещё не начинали.
Когда работа по PR закончена, этот раздел можно удалить.

### Как продолжить локально

Заметки живут в ветке `claude/admiring-maxwell-dgg6lu`: она от `main-vxcontrol` и содержит только этот файл и
`.claude/reviews/pagi-125/`. Правки делаются в ветке PR `fix/PAGI-125-release-review-findings`. Чтобы заметки не
попали в релиз, держите их в отдельном worktree и подключайте к Claude Code импортом:

```sh
git fetch origin
# заметки — отдельным worktree рядом с клоном
git worktree add --track -b claude/admiring-maxwell-dgg6lu ../langchaingo-review-notes origin/claude/admiring-maxwell-dgg6lu
# правки — в ветке PR (или в своей ветке от неё)
git switch fix/PAGI-125-release-review-findings
# подключить заметки к Claude Code, не коммитя их
printf '@../langchaingo-review-notes/CLAUDE.md\n' > CLAUDE.md
echo 'CLAUDE.md' >> .git/info/exclude
```

При первом запуске Claude Code спросит разрешение на внешний импорт. Пути `.claude/reviews/pagi-125/...` ниже —
внутри worktree заметок, то есть `../langchaingo-review-notes/.claude/reviews/pagi-125/...`. Новые заметки
коммитить в worktree заметок и пушить в `claude/admiring-maxwell-dgg6lu`.

### Что ревьюим

- PR: https://github.com/vxcontrol/langchaingo/pull/1, ветка `fix/PAGI-125-release-review-findings` →
  `main-vxcontrol`. Автор PR — сам пользователь (`sirozha`).
- База — предыдущий релиз: `origin/main-vxcontrol` @ `da7016e`. Это ровно merge-base, поэтому
  `git diff origin/main-vxcontrol...origin/fix/PAGI-125-release-review-findings` совпадает с diff'ом сквош-коммита и с
  вкладкой «Files changed» в PR. По коммитам не ревьюим: их 816, многие правят предыдущие коммиты той же ветки.
- Оба ревью сделаны на голове `355dc71`. Если голова сдвинулась, новые изменения смотреть отдельно:
  `git diff 355dc71..<голова>`, и проверять, какие треды закрыты правками.

Объём diff'а:

| Что | Файлов | Строк |
|---|---|---|
| Go-код (не тесты) | 130 | +9 262 / −2 863 |
| Go-тесты | 306 | +31 399 / −3 298 |
| Документация и прочее | 27 | ~+180 |
| Кассеты `testdata/*.httprr` | 225 | +214k (~7,7 МБ) — модели не отдавать, проверять скриптами |

### Что уже сделано

1. **Первое ревью** — `/code-review max --comment`, один агент, только код без тестов:
   [15 комментариев](https://github.com/vxcontrol/langchaingo/pull/1#pullrequestreview-5334862871).
2. **Второе ревью (ultra)** — многоагентное: 4 параллельных workflow, 149 агентов, ~90 минут:
   [87 комментариев](https://github.com/vxcontrol/langchaingo/pull/1#pullrequestreview-5336758859)
   (3 high, 46 medium, 34 low, 4 nit). В PR сейчас 102 треда.
   - 14 частей: 8 частей кода (каждая в два прохода), 4 части тестов, тестовая инфраструктура и кассеты, сквозные
     проверки (apidiff, vet, golangci-lint, `-race`, go mod, примеры, документация).
   - Каждое замечание проверяли два агента: один воспроизводил пробным тестом на HEAD и базе, второй пытался
     опровергнуть; при расхождении решал третий. 117 найдено → 106 подтверждено → 87 после слияния повторов.
   - В общем тексте ревью перечислены ломающие изменения API для release notes и итог перепроверки первого ревью.
3. **Проверено и чисто на `355dc71`:** CI (80 из 80 проверок), `go vet ./...`, `golangci-lint` v2.12.2 (версия из
   CI), `go test -race` (гонок нет), `go mod tidy -diff`, сборка всех 75 примеров. Все изменённые кассеты парсятся и
   воспроизводятся, осиротевших нет, в изменённых кассетах только заглушка `test-api-key`.
4. **Замечания отсортированы** (`triage.md`) и все материалы ревью сохранены в ветке заметок.

Стоимость (эквивалент по прайсу API, Opus 5.5; по подписке это расход лимитов, а не деньги):

| Прогон | ≈ $ |
|---|---|
| `/code-review max` | 18 |
| Ultra: 4 workflow, 149 агентов | 212 |
| Ultra: подготовка, сведение, публикация 87 комментариев | 15 |
| Прочее: анализ, сравнение ревью, сортировка, заметки | 12 |
| **Вся облачная сессия** | **257** |

### Главные темы второго ревью

- **Правила по имени модели без учёта хоста** (OpenRouter, vLLM, Groq, Together): `llms/reasoning/effort_wire.go`
  строки 29, 49, 76, 180, 211, 301; `llms/reasoning/off.go` 171, 195; `llms/reasoning/claude_capability.go:49`.
  Тот же корень, что у первых замечаний #2, #5, #6. Образец исправления: `ServedByDeepSeek(model, host)`,
  `DashScopeRoute(model, host)`.
- **googleai/vertex на GenAI SDK ломает существующих вызывающих Vertex:** credentials + API key (и ключ в тексте
  ошибки), `WithEndpoint` в виде `host:port`, пропал регион по умолчанию, `ClientOptions` игнорируются,
  `WithHTTPClient` теряет авторизацию, удалены экспортируемые идентификаторы.
- **Тесты, которые в CI не запускаются или не могут упасть:** новые интеграционные тесты pgvector, облачные тесты
  ollama и несколько тестов с заведомо проходящими проверками.
- **Три high:** `effort_wire.go:76` (temperature уходит вместе с reasoning_effort для Azure-деплойментов и
  неклассифицированных GPT-5), `effort_wire.go:211` (бюджет для GLM/Kimi/MiniMax отклоняется на любом хосте),
  `off.go:195` (`WithReasoningDisabled` падает для Qwen/QwQ вне DashScope).

### Перепроверка первого ревью (15 замечаний)

12 подтверждены, 3 признаны ложными:

| # | Где | Итог |
|---|---|---|
| 1 | `llms/ollama/ollamallm.go:350` — `think` для моделей без thinking | подтверждено |
| 2 | `llms/openai/structured_output.go:235` — `TakesNoJSONSchema` без хоста | подтверждено |
| 3 | `vectorstores/pgvector/pgvector.go:490` — ключи фильтра только идентификаторы | подтверждено |
| 4 | `internal/toolcall/arguments.go:30` — аргументы `null` отклоняются | подтверждено (зависит от вендора) |
| 5 | `llms/openai/openaillm.go:311` — правила Qwen/DashScope на любом хосте | подтверждено |
| 6 | `llms/reasoning/effort_wire.go:12` — `RejectsPenalties` без хоста | подтверждено |
| 7 | `llms/googleai/option.go:52` — лимит вывода 16384 | **ложное**: намеренно (476f3a6), модели 8k отключены |
| 8 | `llms/googleai/googleai.go:1096` — penalty во все Gemini | подтверждено (зависит от вендора) |
| 9 | `llms/reasoning/claude_capability.go:285` — нет `-latest` в `noPrefillClaude` | подтверждено |
| 10 | `llms/reasoning_support.go:115` — effort caps не только для `ProviderOpenAI` | подтверждено |
| 11 | `llms/anthropic/anthropicllm.go:922` — tool-сообщение с доп. частями падает | подтверждено |
| 12 | `vectorstores/pgvector/metadata_index.go:191` — `ANALYZE` на каждом старте | подтверждено (второе ревью расширило: блокирует запись) |
| 13 | `llms/ollama/ollamallm.go:358` — дублирование gpt-oss | **ложное**: расхождение недостижимо |
| 14 | `llms/options.go:130` — комментарий `GetEffort` не на месте | подтверждено |
| 15 | `embeddings/jina/options.go:86` — комментарий vs `BatchSize` | **ложное**: комментарий верен; рядом настоящий баг — panic при `WithBatchSize(0)` (строка 87) |

### Материалы ревью

`.claude/reviews/pagi-125/` в ветке заметок; подробности — в `README.md` там же:

- `triage.md` — сортировка 99 подтверждённых замечаний на P0/P1/P2 по 10 группам исправлений (A–J).
- `findings.json` — все замечания второго ревью с вердиктами и текстом комментариев, плюс перепроверка первого.
- `not-published.md` — отклонённые замечания, ложные замечания первого ревью, заметки агентов «проверено, но не
  опубликовано» (кандидаты в задачи следующего патча).
- `api-incompat.txt` — 19 несовместимостей публичного API (`apidiff`).
- `probes/` — пробные тесты агентов; из них удобно делать регрессионные тесты.
- `workflow/` — скрипт ultra-ревью и аргументы четырёх запусков, чтобы повторить ревью на изменениях после правок.

### Что сделать перед исправлениями

Критерий из описания PR: изменение, которое отнимает у вызывающего работавшее поведение, — дефект. По нему сделана
сортировка.

1. Утвердить сортировку в `triage.md`: P0 — 39 (блокирует релиз), P1 — 35, P2 — 25 (следующий патч).
2. Закрыть три ложных треда первого ревью с пояснением, чтобы их не «исправили»: #7 — `PRRT_kwDOOj8GYc6mkFBf`,
   #13 — `PRRT_kwDOOj8GYc6mkFV0`, #15 — `PRRT_kwDOOj8GYc6mkFaL` (команда — в разделе «GitHub локально»).
3. Решить, куда идут правки: прямо в `fix/PAGI-125-release-review-findings` или в отдельную ветку от `355dc71` со
   стековым PR в неё. Ветку заметок для правок не использовать.
4. Согласовать общее исправление группы A (13 замечаний: правила по имени модели без хоста) — это ядро
   `llms/reasoning`. Предлагаемый подход: ограничения API вендора применять только на его хосте или маршруте
   LiteLLM, в остальных случаях — поведение базы.
5. Решить по googleai/vertex (группа C): вернуть прежнее поведение или объявить breaking change.
6. Вендор-зависимые замечания (15 из 39 P0): проверить живыми вызовами (ключи, pentagi) или исправлять
   консервативно — возвращать поведение базы там, где правило вендора не измерено.
7. После решений дополнить список breaking changes в описании PR: сейчас в нём 5 пунктов, `apidiff` насчитал 19.

Предлагаемый порядок: P0 по группам A → C → D → H → I, затем P1; каждая группа — отдельной серией коммитов с
регрессионными тестами из `probes/`.

### После исправлений

- Посмотреть `git diff 355dc71..<новая голова>`, пройти по 102 тредам, отметить исправленные, проверить CI.
- Повторное ревью достаточно облегчённое: только новые изменения, одна проверка вместо двух, замечания с общей
  причиной — одним комментарием.

### Окружение для тестов

- Bedrock: задать заглушки `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` и убрать `AWS_CA_BUNDLE` и
  `AWS_PROFILE` — иначе воспроизведение кассет падает до поиска записи (замечание в `bedrockllm_test.go:54`).
- Новые интеграционные тесты pgvector (`narrowing_test.go`, `init_test.go`) запускаются только с
  `PGVECTOR_CONNECTION_STRING`, остальной pgvector и другие vectorstores — через testcontainers (нужен Docker).
- Облачные тесты ollama пропускаются без `~/.ollama/id_ed25519`; для воспроизведения подходит любой ключ.
- `TestCountTokens`, `chains` и `chains/constitution` требуют сети.
- В облачной песочнице Docker не было, поэтому Docker-тесты при ревью не запускались — локально их стоит прогнать.

### GitHub локально

Треды читать и закрывать через `gh` (нужна авторизация):

```sh
# все inline-комментарии PR
gh api repos/vxcontrol/langchaingo/pulls/1/comments --paginate --jq '.[] | "\(.path):\(.line) \(.body[0:80])"'
# id тредов для закрытия (тредов больше 100, --paginate листает страницы)
gh api graphql --paginate -f query='query($endCursor:String){repository(owner:"vxcontrol",name:"langchaingo"){pullRequest(number:1){reviewThreads(first:100,after:$endCursor){pageInfo{hasNextPage endCursor}nodes{id isResolved comments(first:1){nodes{path line}}}}}}}'
# закрыть тред
gh api graphql -f query='mutation($id:ID!){resolveReviewThread(input:{threadId:$id}){thread{isResolved}}}' -f id=PRRT_...
```

### Как запускать ревью

- `/code-review` выполняется форком без Agent tool, то есть одним агентом. Для параллельного ревью нужен Workflow;
  многоагентный запуск — только с явного согласия пользователя в текущей сессии.
- Скрипт ultra-ревью — `.claude/reviews/pagi-125/workflow/pr1-ultra-review.js`. В аргументах заменить `{{SCRATCH}}`
  на свой каталог с рабочими копиями HEAD и базы (`git worktree add --detach`) и поправить блок про окружение в `ctx`.
  В облачном контейнере было 4 CPU, поэтому один workflow держал только 2 агента и запускались 4 workflow параллельно.
- Пробные тесты — через `go test -overlay`: файл пробы лежит вне репозитория, рабочие копии не меняются.
- Якоря inline-комментариев должны попадать в строки diff'а (добавленные или контекст хунка с 3 строками).
