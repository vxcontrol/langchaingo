package reasoning

import (
	"cmp"
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAClaudeVersionTheTablesDoNotListFollowsTheNewestOneBelowIt(t *testing.T) {
	t.Parallel()

	for name, want := range map[string]string{
		"claude-opus-6":                   "claude-opus-5-5",
		"claude-opus-5-6":                 "claude-opus-5-5",
		"claude-opus-4-10":                "claude-opus-4-8",
		"claude-opus-4-9":                 "claude-opus-4-8",
		"claude-sonnet-6":                 "claude-sonnet-5-5",
		"claude-fable-6":                  "claude-fable-5-1",
		"claude-haiku-5":                  "claude-haiku-4-5",
		"claude-haiku-4-6":                "claude-haiku-4-5",
		"us.anthropic.claude-opus-6-v1:0": "claude-opus-5-5",
		"claude-opus-6@20270101":          "claude-opus-5-5",
	} {
		got := canonicalClaude(name)
		require.Contains(t, got, want, name)
		require.Equal(t, ClaudeReasoningKindFor(want), ClaudeReasoningKindFor(name), name)
	}

	for _, documented := range []string{
		"claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-opus-5-5", "claude-sonnet-4-5-20250929",
		"anthropic.claude-haiku-4-5-20251001-v1:0", "claude-opus-latest", "claude-mythos-preview",
		"claude-3-5-haiku-20241022", "claude-opus-3",
	} {
		_, inherited := inheritClaude(canonicalClaude(documented))
		require.False(t, inherited, documented)
	}
}

func TestEveryClaudeVersionATableKeysOnIsAListedRelease(t *testing.T) {
	t.Parallel()

	tables := slices.Concat([][]string{
		adaptiveOnlyClaude, dualClaude, budgetOnlyClaude, alwaysOnClaude, betweenToolsOffClaude, defaultOnClaude,
		noPrefillClaude, noForcedToolClaude, mutuallyExclusiveSamplingClaude, legacyNoStructuredClaude,
		bedrockStructuredClaude, preAdaptiveClaude, rejectsSamplingClaude, budgetEffortClaude,
		bedrockRejectsBudgetEffortClaude, budgetInterleavingClaude,
	}, slices.Collect(maps.Values(bedrockTopEfforts)))
	for _, table := range tables {
		for _, entry := range table {
			if strings.HasSuffix(entry, "-20") {
				continue
			}
			documented, inherited := inheritClaude(entry)
			require.False(t, inherited, "%s is keyed on by a table, yet would be read as %s", entry, documented)
			newest, alias := latestClaude(entry)
			require.False(t, alias, "%s is keyed on by a table, yet is read as %s", entry, newest)
		}
	}
}

func TestAClaudeLatestAliasAnswersEveryTableLikeTheNewestReleaseOfItsTier(t *testing.T) {
	t.Parallel()

	for tier, line := range claudeReleases {
		newest := slices.MaxFunc(line, compareVersions).members[""]
		for _, frame := range []string{"", "~anthropic/", "anthropic/"} {
			alias := frame + "claude-" + tier + "-latest"
			got, want := tableAnswers(alias), tableAnswers(frame+newest)
			for name, rule := range map[string]func(string) bool{
				"ClaudeMutuallyExclusiveSampling": ClaudeMutuallyExclusiveSampling,
				"ClaudeSupportsStructuredOutput":  ClaudeSupportsStructuredOutput,
				"ClaudePredatesAdaptive":          ClaudePredatesAdaptive,
				"ClaudeInterleavesOnBudget":       ClaudeInterleavesOnBudget,
				"ClaudeSpendsThinkingBudget":      ClaudeSpendsThinkingBudget,
			} {
				got[name], want[name] = rule(alias), rule(frame+newest)
			}
			for key := range want {
				assert.Equal(t, want[key], got[key], "%s should answer %s like %s", alias, key, newest)
			}
			_, inherited := InheritedModel(alias)
			assert.False(t, inherited, "%s names a listed release", alias)
		}
	}
}

func tableAnswers(model string) map[string]any {
	answers := map[string]any{}
	for name, rule := range map[string]func(string) bool{
		"AcceptsEffortWire": AcceptsEffortWire, "ChatCompletionsUnsupported": ChatCompletionsUnsupported,
		"ChatToolsUnsupported": ChatToolsUnsupported, "ClaudeRejectsAssistantPrefill": ClaudeRejectsAssistantPrefill,
		"ClaudeRejectsForcedToolUse": ClaudeRejectsForcedToolUse, "ClaudeRejectsSampling": ClaudeRejectsSampling,
		"ClaudeSupportsThinking": ClaudeSupportsThinking, "ClaudeThinkingAlwaysOn": ClaudeThinkingAlwaysOn,
		"ClaudeThinkingDefaultsOn": ClaudeThinkingDefaultsOn, "ClaudeTurnsOffBetweenTools": ClaudeTurnsOffBetweenTools,
		"DashScopeBudgetSharesAnswerLimit": DashScopeBudgetSharesAnswerLimit, "DashScopeTakesNoTopK": DashScopeTakesNoTopK,
		"DashScopeTakesThinkingBudget": DashScopeTakesThinkingBudget, "FixesSampling": FixesSampling,
		"GeminiCanDisable": GeminiCanDisable, "GeminiSupportsThinking": GeminiSupportsThinking,
		"GeminiUsesThinkingLevel": GeminiUsesThinkingLevel, "GrokFamily": GrokFamily, "IsReasoningModel": IsReasoningModel,
		"LikelyReasoningModel": LikelyReasoningModel, "OpenAIThinkingOptIn": OpenAIThinkingOptIn,
		"QwenThinkingEnabledByFlag": QwenThinkingEnabledByFlag, "QwenThinkingOffByFlag": QwenThinkingOffByFlag,
		"QwenThinkingRequiresStream": QwenThinkingRequiresStream, "RejectsMinP": RejectsMinP,
		"RejectsPenalties": RejectsPenalties, "RejectsRepetitionPenalty": RejectsRepetitionPenalty,
		"RejectsSamplingWhileThinking": RejectsSamplingWhileThinking, "RejectsStop": RejectsStop,
		"RejectsTopK": RejectsTopK, "ReplaysEmptyReasoning": ReplaysEmptyReasoning,
		"ReplaysReasoningOnEveryTurn": ReplaysReasoningOnEveryTurn, "ReplaysThinkingInContent": ReplaysThinkingInContent,
		"ServedByMistral": ServedByMistral, "TakesNoJSONObject": TakesNoJSONObject,
		"TakesNoResponseFormat": TakesNoResponseFormat, "TakesNoThinkingDepth": TakesNoThinkingDepth,
		"TakesNoTopK": TakesNoTopK, "ThinkingOptIn": ThinkingOptIn,
	} {
		answers[name] = rule(model)
	}
	answers["ClaudeReasoningKindFor"] = ClaudeReasoningKindFor(model)
	answers["EffortWithTools"] = EffortWithTools(model)
	answers["OpenAIReasoningCapsFor"] = OpenAIReasoningCapsFor(model)
	answers["GeminiThinkingLevels"] = GeminiThinkingLevels(model)
	for _, p := range []Provider{ProviderOpenAI, ProviderAnthropic, ProviderBedrock, ProviderGoogleAI, ProviderOllama} {
		answers[fmt.Sprint("ResolveOff/", p)] = ResolveOff(model, p)
		answers[fmt.Sprint("ClaudeEffortsFor/", p)] = ClaudeEffortsFor(model, p)
	}
	for _, host := range []string{
		"api.deepseek.com", "api.z.ai", "api.moonshot.ai", "dashscope-intl.aliyuncs.com", "api.x.ai", "api.openai.com",
		"api.mistral.ai", "litellm.internal", "openrouter.ai",
	} {
		answers["ServedBy/"+host] = ServedBy(model, host)
		answers["TakesNoJSONSchema/"+host] = TakesNoJSONSchema(model, host)
		answers["UsesLegacyMaxTokens/"+host] = UsesLegacyMaxTokens(model, host)
		answers["ServedByDeepSeek/"+host] = ServedByDeepSeek(model, host)
		answers["ServedByZAI/"+host] = ServedByZAI(model, host)
		answers["RejectsRequiredToolChoice/"+host] = RejectsRequiredToolChoice(model, host)
		answers["RejectsForcedToolChoiceWhileThinking/"+host] = RejectsForcedToolChoiceWhileThinking(model, host, true)
	}
	return answers
}

func TestAnUnlistedVersionAnswersEveryTableLikeTheReleaseItFollows(t *testing.T) {
	t.Parallel()

	for name, documented := range map[string]string{
		"gpt-5.7": "gpt-5.6", "gpt-6.1": "gpt-6.1-sol", "gpt-6.2-sol": "gpt-6.1-sol", "gpt-6.2-astra": "gpt-6-astra",
		"gpt-5.7-pro": "gpt-5.5-pro", "openai/gpt-5.7": "gpt-5.6",
		"glm-5.4": "glm-5.3", "glm-6": "glm-5.3", "glm-5.31": "glm-5.3", "glm-5.4-flash": "glm-5.3-flash",
		"zai-glm-5-4": "zai-glm-5-3", "zai-glm-5": "zai-glm-5-3", "zai-glm-latest": "zai-glm-5-3",
		"kimi-k4": "kimi-k3", "kimi-k2.8": "kimi-k2.6", "moonshot/kimi-k4": "moonshot/kimi-k3",
		"deepseek-v5": "deepseek-flash", "deepseek-v5-pro": "deepseek-flash", "deepseek/deepseek-v5": "deepseek/deepseek-flash",
		"minimax-m4": "minimax-m3", "MiniMax-M4": "MiniMax-M3",
		"qwen3.9-max": "qwen3.8-max", "qwen4-max": "qwen3.8-max", "dashscope/qwen3.9-max": "dashscope/qwen3.8-max",
		"grok-4.8": "grok-4.7", "grok-5": "grok-4.7",
		"claude-opus-6": "claude-opus-5-5", "claude-opus-4-10": "claude-opus-4-8",
		"gemini-4-pro": "gemini-3.1-pro-preview", "gemini-4-flash": "gemini-3.8-flash",
		"models/gemini-4-pro": "gemini-3.1-pro-preview", "gemini-pro-latest": "gemini-3.1-pro-preview",
		"gemini-3.9-flash-lite": "gemini-3.5-flash-lite", "gemma-5": "gemma-4-26b-a4b-it",
	} {
		_, release := splitModelName(documented)
		followed, inherited := InheritedModel(name)
		if !inherited {
			_, followed = splitModelName(name)
		}
		assert.Equal(t, release, followed, "%s should follow %s", name, documented)
		got, want := tableAnswers(name), tableAnswers(documented)
		for key := range want {
			assert.Equal(t, want[key], got[key], "%s should answer %s like %s", name, key, documented)
		}
	}
}

func TestEveryListedReleaseReadsAsItself(t *testing.T) {
	t.Parallel()

	for _, f := range lineFamilies {
		for _, line := range f.lines {
			for _, g := range line {
				for _, id := range g.members {
					documented, inherited := inheritLine(id)
					require.False(t, inherited, "%s reads as %s", id, documented)
					for _, stage := range f.stages {
						released := strings.Replace(id, "-"+stage, "", 1)
						documented, inherited := inheritLine(released)
						require.False(t, inherited, "%s, the release of %s, reads as %s", released, id, documented)
					}
				}
			}
		}
	}
	for _, releases := range claudeReleases {
		for _, r := range releases {
			documented, inherited := inheritClaude(r.members[""])
			require.False(t, inherited, "%s reads as %s", r.members[""], documented)
		}
	}
}

func TestADocumentedQwenAliasOrSnapshotAnswersAsItsStableID(t *testing.T) {
	t.Parallel()

	for name, stable := range map[string]string{
		"qwen-plus-latest": "qwen-plus", "qwen-plus-2025-12-01": "qwen-plus", "qwen-plus-2025-04-28": "qwen-plus",
		"qwen-flash-2025-07-28": "qwen-flash", "qwen-turbo-latest": "qwen-turbo",
		"qwen3-max-2026-01-23": "qwen3-max", "qwen3-max-preview": "qwen3-max",
		"qwen3-vl-plus-2025-12-19": "qwen3-vl-plus", "dashscope/qwen-plus-latest": "dashscope/qwen-plus",
		"qwen3.7-max-2026-05-20": "qwen3.7-max",
	} {
		got, want := tableAnswers(name), tableAnswers(stable)
		for key := range want {
			assert.Equal(t, want[key], got[key], "%s should answer %s like %s", name, key, stable)
		}
		_, inherited := InheritedModel(name)
		assert.False(t, inherited, "%s is documented", name)
	}

	for _, own := range []string{"qwen-plus-2025-01-25", "qwen3-max-2025-09-23"} {
		require.False(t, QwenThinkingEnabledByFlag(own), "%s predates the stable id's thinking", own)
		require.False(t, IsReasoningModel(own), own)
	}
	require.Equal(t, OffUnsupported, ResolveOff("qwen3.7-max-2026-05-17", ProviderOpenAI), "a thinking-only snapshot")
	require.Equal(t, ResolveOff("qwen3.7-max", ProviderOpenAI), ResolveOff("qwen3.7-max-2026-05-20", ProviderOpenAI))
}

func TestAVersionIsReadTheWayItsVendorWritesIt(t *testing.T) {
	t.Parallel()

	for name, want := range map[string]string{
		"grok-4.35": "grok-4.3", "grok-4.65": "grok-4.6", "grok-4.8": "grok-4.7", "grok-4.8-latest": "grok-4.7",
		"kimi-k4:free": "kimi-k3", "moonshotai/kimi-k4:free": "kimi-k3", "kimi-k4:cloud": "kimi-k3",
		"anthropic/claude-sonnet-6:nitro": "claude-sonnet-5-5", "anthropic/claude-opus-4.9:online": "claude-opus-4-8",
		"gemini-4-flash-preview-05-20": "gemini-3.8-flash", "gemini-4-flash-001": "gemini-3.8-flash",
	} {
		documented, inherited := InheritedModel(name)
		assert.True(t, inherited, name)
		assert.Equal(t, want, documented, name)
	}
	for _, own := range []string{
		"grok-4.25", "grok-4.20", "grok-4.30", "grok-4.7-latest", "kimi-k3:free", "gemini-2.0-flash-001",
		"qwen3:32b", "deepseek-v3.1:671b-cloud", "qwen3:30b-a3b", "qwen3:8b-q4_K_M", "qwen3.5:397b-a17b",
		"qwen3", "qwen3.5", "qwen3:latest", "qwen3.9",
	} {
		documented, inherited := InheritedModel(own)
		assert.False(t, inherited, "%s reads as %s", own, documented)
	}
}

func TestTheReleasesBelowAreWrittenInTheCallersSpelling(t *testing.T) {
	t.Parallel()

	require.Equal(t, []string{
		"zai-glm-5.3", "zai-glm-5.2", "zai-glm-5.1", "zai-glm-5", "zai-glm-4.7", "zai-glm-4.6", "zai-glm-4.5",
	}, releasesBelow("zai-glm-6"))
	require.Equal(t, []string{"glm-5-3", "glm-5-2", "glm-5-1", "glm-5", "glm-4-7", "glm-4-6", "glm-4-5"},
		releasesBelow("glm-5-4"))
	require.Equal(t, []string{
		"us.anthropic.claude-opus-5-5", "us.anthropic.claude-opus-5", "us.anthropic.claude-opus-4-8",
		"us.anthropic.claude-opus-4-7", "us.anthropic.claude-opus-4-6", "us.anthropic.claude-opus-4-5",
		"us.anthropic.claude-opus-4-1", "us.anthropic.claude-opus-4-0",
	}, releasesBelow("us.anthropic.claude-opus-6-v1:0"))
	require.Equal(t, OffOmit, InheritedOffWire("zai-glm-6", ProviderOpenAI), "Mistral turns GLM thinking off by omission")
	require.Equal(t, OffDisableThinkingObject, InheritedOffWire("glm-6", ProviderOpenAI))
}

func TestEveryWordALineReadsChangesWhatTheNameFollows(t *testing.T) {
	t.Parallel()

	for name, want := range map[string]string{
		"gemini-3.2-pro-preview-customtools": "gemini-3.1-pro-preview-customtools",
		"gemini-4-pro-preview":               "gemini-3.1-pro-preview",
		"gemini-3.9-flash-lite":              "gemini-3.5-flash-lite",
		"gemma-5-27b-it":                     "gemma-4-26b-a4b-it",
		"gpt-6.2-sol":                        "gpt-6.1-sol",
		"gpt-6.2-luna":                       "gpt-6-luna",
		"gpt-5.7-terra":                      "gpt-5.6-terra",
		"gpt-6.1-astra":                      "gpt-6-astra",
		"gpt-5.7-cyber":                      "gpt-5.6-cyber",
		"gpt-5.6-pro":                        "gpt-5.5-pro",
		"gpt-5.7-mini":                       "gpt-5.6",
		"gpt-5.7-nano":                       "gpt-5.6",
		"glm-5.4-air":                        "glm-5.3",
		"glm-5.4-airx":                       "glm-5.3",
		"glm-5.4-x":                          "glm-5.3",
		"glm-5.4-flash":                      "glm-5.3-flash",
		"glm-5.4-flashx":                     "glm-5.3-flashx",
		"glm-5.4-turbo":                      "glm-5.3",
		"glm-5.4-fast":                       "glm-5.3",
		"glm-5.4-prime":                      "glm-5.3",
		"glm-5.4-preview":                    "glm-5.3",
		"kimi-k2.1-thinking":                 "kimi-k2-thinking",
		"kimi-k2.8-code":                     "kimi-k2.7-code",
		"kimi-k3.1-highspeed":                "kimi-k3",
		"kimi-k3.1-preview":                  "kimi-k3",
		"deepseek-v4.2-flash":                "deepseek-v4.1-flash",
		"deepseek-v4.2-pro":                  "deepseek-flash",
		"deepseek-v4.2-preview":              "deepseek-flash",
		"deepseek-v4.2-exp":                  "deepseek-flash",
		"minimax-m3.2-flash":                 "minimax-m3.1-flash-preview",
		"minimax-m2.8-highspeed":             "minimax-m2.7",
		"minimax-m2.8-stable":                "minimax-m2.7",
		"minimax-m3.2-preview":               "minimax-m3",
		"qwen3.9-max":                        "qwen3.8-max",
		"qwen3.9-plus":                       "qwen3.8-max",
		"qwen3.9-flash":                      "qwen3.8-flash",
		"qwen3.9-turbo":                      "qwen3.8-max",
		"qwen3.9-max-preview":                "qwen3.8-max",
	} {
		documented, inherited := InheritedModel(name)
		assert.True(t, inherited, name)
		assert.Equal(t, want, documented, name)
	}
}

func TestAHiddenNewestReleaseReadsAsTheOneBeforeItInItsOwnLine(t *testing.T) {
	t.Parallel()

	byVersion := func(a, b generation) int {
		return cmp.Or(cmp.Compare(a.major, b.major), cmp.Compare(a.minor, b.minor))
	}
	without := func(line []generation, hidden generation) []generation {
		return slices.DeleteFunc(slices.Clone(line), func(g generation) bool { return byVersion(g, hidden) == 0 })
	}
	for _, f := range lineFamilies {
		for product, line := range f.lines {
			if len(line) < 2 {
				continue
			}
			newest := slices.MaxFunc(line, byVersion)
			hidden := f
			hidden.lines = maps.Clone(f.lines)
			hidden.lines[product] = without(line, newest)
			previous := slices.MaxFunc(hidden.lines[product], byVersion)
			for qualifier, id := range newest.members {
				if p, ok := f.parse(id); !ok || p.product != product {
					continue
				}
				documented, inherited := hidden.inherit(id)
				require.True(t, inherited, "%s with its release hidden", id)
				require.Equal(t, previous.member(qualifier), documented, "%s with its release hidden", id)
			}
		}
	}
	for tier, line := range claudeReleases {
		if len(line) < 2 {
			continue
		}
		newest := slices.MaxFunc(line, byVersion)
		_, major, minor, ok := claudeVersion(newest.members[""])
		require.True(t, ok, tier)
		g, listed, found := nearest(without(line, newest), major, minor)
		require.True(t, found && !listed, tier)
		require.Equal(t, slices.MaxFunc(without(line, newest), byVersion).members[""], g.members[""], tier)
	}
}
