export const meta = {
  name: 'pr1-ultra-review',
  description: 'Ultra review of PR #1 slices: find defects per slice, second-pass gap hunt, two-lens adversarial verification',
  whenToUse: 'Release review of vxcontrol/langchaingo PR #1, parameterized by slices in args',
  phases: [
    { title: 'Find', detail: 'one finder per slice, then a gap-hunting second pass on source slices' },
    { title: 'Verify', detail: 'each finding checked by a reproduce lens and a refute lens' },
    { title: 'Tiebreak', detail: 'a judge settles findings where the two lenses disagree' },
    { title: 'Existing', detail: 'findings of the earlier review re-verified the same way' },
  ],
}

const CTX = args.ctx
const EXISTING_TEXT = args.existing_all
  .map(e => `#${e.n} ${e.file}:${e.line} — ${e.summary} Failure: ${e.failure_scenario}`)
  .join('\n')

const SEVERITIES = ['critical', 'high', 'medium', 'low', 'nit']
const CATEGORIES = ['correctness', 'regression', 'api-compat', 'security', 'concurrency', 'performance', 'error-handling', 'test-quality', 'docs-mismatch', 'cleanup']

const FINDING = {
  type: 'object',
  properties: {
    file: { type: 'string', description: 'repo-relative path' },
    line: { type: 'integer', description: 'line in the HEAD version of the file, inside a hunk of the PR diff' },
    title: { type: 'string' },
    category: { type: 'string', enum: CATEGORIES },
    severity: { type: 'string', enum: SEVERITIES },
    summary: { type: 'string', description: 'one sentence: the defect' },
    failure_scenario: { type: 'string', description: 'concrete inputs/state -> wrong output/crash' },
    evidence: { type: 'string', description: 'what you ran or read to establish it: probe test and its output, code refs, vendor doc URL' },
    deliberate: { type: 'boolean', description: 'true if commit messages/tests show the behavior change was intended' },
    vendor_dependent: { type: 'boolean', description: 'true if the harm depends on remote vendor behavior not reproducible here' },
  },
  required: ['file', 'line', 'title', 'category', 'severity', 'summary', 'failure_scenario', 'evidence', 'deliberate', 'vendor_dependent'],
}
const FINDINGS = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: FINDING },
    coverage_notes: { type: 'string', description: 'what you covered, and what you could not cover and why' },
  },
  required: ['findings', 'coverage_notes'],
}
const REPRO = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          index: { type: 'integer' },
          verdict: { type: 'string', enum: ['confirmed', 'plausible', 'refuted', 'duplicate_existing'] },
          evidence: { type: 'string' },
          anchor_file: { type: 'string' },
          anchor_line: { type: 'integer' },
          severity: { type: 'string', enum: SEVERITIES },
          comment: { type: 'string', description: 'ready-to-post PR comment in English; empty if refuted or duplicate' },
        },
        required: ['index', 'verdict', 'evidence', 'anchor_file', 'anchor_line', 'severity', 'comment'],
      },
    },
  },
  required: ['verdicts'],
}
const REFUTE = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          index: { type: 'integer' },
          verdict: { type: 'string', enum: ['stands', 'refuted', 'duplicate_existing'] },
          preexisting: { type: 'boolean', description: 'true if BASE already behaved the same way' },
          reason: { type: 'string' },
        },
        required: ['index', 'verdict', 'preexisting', 'reason'],
      },
    },
  },
  required: ['verdicts'],
}
const JUDGE = {
  type: 'object',
  properties: {
    keep: { type: 'boolean' },
    severity: { type: 'string', enum: SEVERITIES },
    reason: { type: 'string' },
    anchor_file: { type: 'string' },
    anchor_line: { type: 'integer' },
    comment: { type: 'string', description: 'ready-to-post PR comment in English; empty if not kept' },
  },
  required: ['keep', 'severity', 'reason', 'anchor_file', 'anchor_line', 'comment'],
}

const SOURCE_RULES = `Read the full net diff of your slice (non-test .go files) and the HEAD version of every changed file. For each changed function read its callers and the code it calls, also outside your slice, and compare with BASE behaviour. Read the commit messages that touch each file to learn the intent. Look for: logic errors and wrong conditions; nil/empty/zero-value handling; error paths that lose, swallow or misreport errors; regressions for existing callers (a call that worked on BASE and now fails or behaves differently without a documented reason); inconsistent handling of the same concept between provider doors; rules keyed on the wrong input (e.g. model name without host/provider); model-name normalization gaps (vendor/region/profile prefixes, date or version suffixes, -latest aliases, case, ':tag' suffixes); concurrency problems (shared maps, goroutines, streaming); resource leaks; SQL or identifier injection; comments and docs in the diff that claim something the code does not do. Look at the slice's tests too when they reveal a source bug (a test pinning wrong behaviour). Confirm findings with probe tests where feasible.`

const TEST_RULES = `Review the changed *_test.go files of your slice (net diff) for test defects: tests that cannot fail (no assertions, asserting on a value they just set, comparing a value with itself, ignored errors); assertions that check the wrong thing or encode behaviour that contradicts the production code's contract or vendor documentation; table-driven tests whose cases are never run or hidden; tests that are not executed at all (wrong name, build tags, missing t.Run); skip conditions that silently hide failures in CI; httprr cassettes that do not match what the test sends; t.Parallel with shared global state or env vars; goroutine leaks; flakiness (time, map order, randomness). Run the slice's tests on HEAD (go test -p 2 ./<pkg>/...; Bedrock with env -u AWS_CA_BUNDLE) and report real failures. Anchor findings to the _test.go line. Report a missing test only when a risky production change in the slice has no test at all, and name the untested behaviour.`

const INFRA_RULES = `Review the test infrastructure changes (testing/llmtest, internal/httprr, internal/devtools, internal/testutil) for defects that make tests pass when they should fail or fail spuriously: record/replay matching, request body handling, credential scrubbing (keys must never be written to cassettes), credential checks that skip tests, helpers that swallow errors. Then check cassette integrity with scripts, without reading cassette bodies into your context: every testdata/*.httprr file added or changed in the PR is used by some test (find the file-naming convention in internal/httprr), and every test that opens a cassette has one; report orphans and missing ones. Scan the cassettes for credentials other than the placeholder test-api-key (Authorization, X-Api-Key, X-Goog-Api-Key, x-amz-*, api-key headers, key=/token= query parameters, cookies) and for personal data. Anchor findings to a line of a changed .go file where possible.`

const CROSS_RULES = `Cross-cutting checks over the whole PR. (1) Public API compatibility: list exported identifiers (functions, methods, types, struct fields, constants, variables, interface methods) that were removed, renamed or changed signature or documented semantics between BASE and HEAD in every package (script over go doc -all output per package, or install golang.org/x/exp/cmd/apidiff if the module proxy allows); report each break that is not clearly intentional and documented. (2) Run go vet ./... and golangci-lint run ./... (repo config .golangci.yaml) on HEAD and on BASE; report only problems new in HEAD. (3) Run go test -race -p 2 on the changed packages (llms/..., internal/..., chains, embeddings/..., vectorstores/inmemory, vectorstores/pgvector if it runs without Docker; Bedrock with env -u AWS_CA_BUNDLE) and report data races and failures. (4) go.mod/go.sum changes. (5) examples/ and docs/ changes that no longer compile or contradict the code. Report only concrete problems, anchored to a changed line.`

const COMMON_RULES = `## Rules
- Report real defects with concrete impact; no style preferences. Cleanups only when they hide a real hazard (e.g. two copies of a rule that already disagree).
- A behaviour change that the commit history shows as deliberate is not a defect by itself. Report it only if it breaks existing callers in a way the change did not intend, or contradicts vendor documentation; set deliberate=true.
- Give file and line on HEAD. The line must be inside a hunk of the PR diff (an added line or a context line as shown by git diff with 3 lines of context) because it becomes an inline PR comment; pick the most relevant line of the hunk.
- Try to confirm each finding with a probe test and state in evidence what you ran and what it printed. If the harm depends on vendor behaviour you cannot run, set vendor_dependent=true and cite documentation if you found it.
- Be exhaustive within your slice. Return an empty list if you find nothing real.`

function rulesFor(kind) {
  if (kind === 'tests') return TEST_RULES
  if (kind === 'infra') return INFRA_RULES
  if (kind === 'cross') return CROSS_RULES
  return SOURCE_RULES
}

function sliceText(s) {
  return `Slice "${s.name}" (${s.kind}).\nPaths: ${s.paths.join('; ')}\nFocus: ${s.focus}`
}

function findingText(f, i) {
  return `### Finding ${i}\nFile: ${f.file}:${f.line}\nTitle: ${f.title || ''}\nSummary: ${f.summary}\nFailure scenario: ${f.failure_scenario}\nReviewer's evidence: ${f.evidence || '(none)'}`
}

function findPrompt(s) {
  return `${CTX}\n\n## Your slice\n${sliceText(s)}\n\n## How to review\n${rulesFor(s.kind)}\n\n${COMMON_RULES}\n\n## Already reported in an earlier review (do not report again)\n${EXISTING_TEXT}`
}

function gapPrompt(s, prior) {
  const list = prior.length ? prior.map((f, i) => `${i + 1}. ${f.file}:${f.line} — ${f.summary}`).join('\n') : '(none)'
  return `${CTX}\n\n## Your slice\n${sliceText(s)}\n\n## Your task\nThis is a second, independent pass over the same slice. A first reviewer reported the findings listed below. Hunt for defects they missed, from different angles: walk each changed function's error and edge paths one by one; check interactions between the changed files and their callers outside the slice; compare the exported behaviour of every changed exported function with BASE; check every comment and doc statement in the diff against the code; check the slice's tests for behaviour they pin that is actually wrong. Do not repeat the first reviewer's findings or the already-reported ones.\n\n${rulesFor(s.kind)}\n\n${COMMON_RULES}\n\n## First reviewer's findings\n${list}\n\n## Already reported in an earlier review (do not report again)\n${EXISTING_TEXT}`
}

function reproPrompt(batch, isExisting) {
  const note = isExisting
    ? 'These findings were posted in an earlier review and are being re-checked; never answer duplicate_existing.'
    : 'Answer duplicate_existing only if the finding is the same defect at the same place as an already-reported item.'
  return `${CTX}\n\n## Your task\nYou verify review findings by trying to reproduce them. For each finding below: read the code at HEAD yourself and try to demonstrate the failure scenario concretely, preferably with a probe test whose output shows the wrong behaviour at HEAD; for claimed regressions also show how BASE behaves. If the harm depends on remote vendor behaviour, check the vendor's documentation (WebSearch/WebFetch) and cite it. Verdicts: confirmed = demonstrated, or unambiguous from the code; plausible = the code path clearly leads there but depends on external behaviour you could only partly confirm; refuted = the scenario does not happen as described (explain why); duplicate_existing = see below. ${note}\nFor each finding also give the best anchor (file and HEAD line inside a hunk of the PR diff), a severity, and for confirmed/plausible a ready-to-post PR comment in English: 2–5 sentences with the defect, the concrete failure scenario, what you ran to show it, and the fix direction; say explicitly when it depends on vendor behaviour. Use index = the finding number.\n\n${batch.map((f, i) => findingText(f, i + 1)).join('\n\n')}\n\n## Already reported in an earlier review\n${EXISTING_TEXT}`
}

function refutePrompt(batch, isExisting) {
  const note = isExisting
    ? 'These findings were posted in an earlier review and are being re-checked; never answer duplicate_existing.'
    : 'Answer duplicate_existing only if the finding is the same defect at the same place as an already-reported item.'
  return `${CTX}\n\n## Your task\nYou are a skeptical maintainer of this library. For each finding below, look for the strongest reason it is NOT a defect worth fixing: the behaviour is intended (commit messages, tests pinning it on purpose, docs); the input cannot occur in practice; callers already handle it; vendor documentation contradicts the claim. Read the code yourself; do not trust the finding's evidence. Also record whether BASE already behaved the same way (preexisting). Verdicts: stands = you found no solid reason to dismiss it; refuted = it is not a defect (give the reason); duplicate_existing = see below. Answer refuted when the claim rests on assumptions the code or documentation contradicts; if it is merely hard to prove but the code clearly leads there, it stands. ${note} Use index = the finding number.\n\n${batch.map((f, i) => findingText(f, i + 1)).join('\n\n')}\n\n## Already reported in an earlier review\n${EXISTING_TEXT}`
}

function judgePrompt(f, r, x, isExisting) {
  const rs = r ? `verdict=${r.verdict}; evidence: ${r.evidence}` : 'no answer (verifier failed)'
  const xs = x ? `verdict=${x.verdict}; preexisting=${x.preexisting}; reason: ${x.reason}` : 'no answer (verifier failed)'
  const note = isExisting ? 'This finding was posted in an earlier review; decide whether it is real.' : ''
  return `${CTX}\n\n## Your task\nTwo verifiers disagree (or one failed) about the review finding below. ${note} Read the code yourself, run a probe if it helps, and decide whether it is a real defect worth an inline PR comment. Keep it only if you are convinced the failure scenario happens, or, for vendor-dependent claims, documentation supports it. Return keep, severity, reason, the best anchor (file and HEAD line inside a hunk of the PR diff) and, if kept, a ready-to-post PR comment in English (2–5 sentences: the defect, the concrete failure scenario, the evidence, the fix direction).\n\n${findingText(f, 1)}\n\n## Reproduce verifier\n${rs}\n\n## Skeptic verifier\n${xs}\n\n## Already reported in an earlier review\n${EXISTING_TEXT}`
}

function pick(res, index) {
  if (!res || !Array.isArray(res.verdicts)) return null
  return res.verdicts.find(v => v.index === index) || null
}

async function settle(f, r, x, label, isExisting) {
  const rv = r ? r.verdict : 'missing'
  const xv = x ? x.verdict : 'missing'
  const base = { ...f, repro: r, refute: x }
  if (!isExisting && rv === 'duplicate_existing' && xv === 'duplicate_existing') {
    return { ...base, keep: false, decided_by: 'duplicate' }
  }
  const rOk = rv === 'confirmed' || rv === 'plausible'
  const xOk = xv === 'stands'
  if (rOk && xOk) {
    return { ...base, keep: true, decided_by: 'both', severity_final: r.severity, anchor_file: r.anchor_file, anchor_line: r.anchor_line, comment: r.comment, certainty: rv, preexisting: x.preexisting }
  }
  if (rv === 'refuted' && xv === 'refuted') {
    return { ...base, keep: false, decided_by: 'both' }
  }
  const j = await agent(judgePrompt(f, r, x, isExisting), { label: `judge:${label}`, phase: 'Tiebreak', schema: JUDGE })
  if (!j) return { ...base, keep: false, decided_by: 'judge-failed' }
  return { ...base, keep: j.keep, decided_by: 'judge', judge_reason: j.reason, severity_final: j.severity, anchor_file: j.anchor_file, anchor_line: j.anchor_line, comment: j.comment, certainty: 'judged', preexisting: x ? x.preexisting : null }
}

async function verifyBatch(findings, name, isExisting) {
  if (!findings || !findings.length) return []
  const batches = []
  for (let i = 0; i < findings.length; i += 3) batches.push(findings.slice(i, i + 3))
  const phaseName = isExisting ? 'Existing' : 'Verify'
  const perBatch = await parallel(batches.map((b, bi) => async () => {
    const lenses = await parallel([
      () => agent(reproPrompt(b, isExisting), { label: `repro:${name}:${bi + 1}`, phase: phaseName, schema: REPRO, effort: 'high' }),
      () => agent(refutePrompt(b, isExisting), { label: `refute:${name}:${bi + 1}`, phase: phaseName, schema: REFUTE, effort: 'high' }),
    ])
    const rep = lenses[0]
    const ref = lenses[1]
    const settled = await parallel(b.map((f, i) => () => settle(f, pick(rep, i + 1), pick(ref, i + 1), `${name}:${bi + 1}.${i + 1}`, isExisting)))
    return settled
  }))
  return perBatch.filter(Boolean).flat().filter(Boolean)
}

const slices = args.slices
log(`Slices: ${slices.map(s => s.name).join(', ')}; earlier findings to re-verify: ${(args.existing_verify || []).length}`)

const sliceRun = pipeline(
  slices,
  s => agent(findPrompt(s), { label: `find:${s.name}`, phase: 'Find', schema: FINDINGS }),
  async (r1, s) => {
    const f1 = (r1 && r1.findings ? r1.findings : []).map(f => ({ ...f, slice: s.name, pass: 1 }))
    log(`${s.name}: first pass ${r1 ? f1.length + ' findings' : 'FAILED (no result)'}`)
    const both = await parallel([
      () => (s.gap ? agent(gapPrompt(s, f1), { label: `gap:${s.name}`, phase: 'Find', schema: FINDINGS }) : Promise.resolve(null)),
      () => verifyBatch(f1, s.name, false),
    ])
    const gap = both[0]
    const v1 = both[1] || []
    const f2 = (gap && gap.findings ? gap.findings : []).map(f => ({ ...f, slice: s.name, pass: 2 }))
    if (s.gap) log(`${s.name}: second pass ${gap ? f2.length + ' findings' : 'FAILED (no result)'}`)
    const v2 = await verifyBatch(f2, `${s.name}+gap`, false)
    const all = [...v1, ...v2]
    log(`${s.name}: ${all.filter(x => x.keep).length} of ${all.length} findings survived verification`)
    return {
      slice: s.name,
      first_pass_ok: !!r1,
      gap_ok: s.gap ? !!gap : null,
      coverage: r1 ? r1.coverage_notes : null,
      gap_coverage: gap ? gap.coverage_notes : null,
      findings: all,
    }
  },
)

const existingRun = verifyBatch(args.existing_verify || [], 'existing', true)
const done = await Promise.all([sliceRun, existingRun])
const sliceResults = done[0]
sliceResults.forEach((r, i) => { if (!r) log(`slice ${slices[i].name}: pipeline failed, no results`) })
return { slices: sliceResults, existing: done[1] }
