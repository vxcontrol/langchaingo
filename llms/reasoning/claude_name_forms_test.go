package reasoning

import (
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestClaudePredicatesAgreeAcrossNameForms(t *testing.T) {
	t.Parallel()

	for _, group := range [][]string{
		{"claude-opus-4-7", "claude-opus-4.7", "anthropic/claude-opus-4-7", "us.anthropic.claude-opus-4-7-v1:0"},
		{"claude-sonnet-4-5", "claude-sonnet-4.5", "anthropic/claude-sonnet-4.5", "claude-sonnet-4-5@20250514"},
		{"claude-opus-4-6", "claude-opus-4.6", "eu.anthropic.claude-opus-4-6-v1:0"},
		{"claude-haiku-4-5", "claude-haiku-4.5"},
		{"claude-sonnet-5", "claude-sonnet-5.0", "us.anthropic.claude-sonnet-5-v1:0"},
	} {
		canonical := group[0]
		want := struct {
			kind    ClaudeReasoningKind
			reject  bool
			mutex   bool
			prefill bool
		}{
			ClaudeReasoningKindFor(canonical),
			ClaudeRejectsSampling(canonical),
			ClaudeMutuallyExclusiveSampling(canonical),
			ClaudeRejectsAssistantPrefill(canonical),
		}
		if want.kind == ClaudeReasoningUnknown {
			t.Fatalf("%s must be classified for this table to mean anything", canonical)
		}

		for _, form := range group[1:] {
			t.Run(form, func(t *testing.T) {
				t.Parallel()
				if got := ClaudeReasoningKindFor(form); got != want.kind {
					t.Errorf("ClaudeReasoningKindFor = %v, want %v (as %s)", got, want.kind, canonical)
				}
				if got := ClaudeRejectsSampling(form); got != want.reject {
					t.Errorf("ClaudeRejectsSampling = %v, want %v (as %s)", got, want.reject, canonical)
				}
				if got := ClaudeMutuallyExclusiveSampling(form); got != want.mutex {
					t.Errorf("ClaudeMutuallyExclusiveSampling = %v, want %v (as %s)", got, want.mutex, canonical)
				}
				if got := ClaudeRejectsAssistantPrefill(form); got != want.prefill {
					t.Errorf("ClaudeRejectsAssistantPrefill = %v, want %v (as %s)", got, want.prefill, canonical)
				}
			})
		}
	}
}

func TestAClaudeIDSpelledAnotherWayAnswersAsItsRelease(t *testing.T) {
	t.Parallel()

	for form, release := range map[string]string{
		"claude-5.5-opus": "claude-opus-5-5", "claude-5-5-opus": "claude-opus-5-5", "claude-4.7-opus": "claude-opus-4-7",
		"claude-4.6-sonnet": "claude-sonnet-4-6", "claude-5-fable": "claude-fable-5", "claude-5.1-fable": "claude-fable-5-1",
		"claude-5-mythos": "claude-mythos-5", "claude-sonnet-4": "claude-sonnet-4-0", "claude-opus-4": "claude-opus-4-0",
		"claude-4-sonnet": "claude-sonnet-4-0", "claude-4-opus-20250514": "claude-opus-4-0",
		"claude-opus-4-6[1m]": "claude-opus-4-6", "claude-opus-5-5[1m]": "claude-opus-5-5", "claude-sonnet-4-5[1m]": "claude-sonnet-4-5",
		"claude-sonnet-4-5-20250929[1m]": "claude-sonnet-4-5", "claude-opus-5[1m]": "claude-opus-5", "claude-sonnet-5[1m]": "claude-sonnet-5",
	} {
		for _, frame := range []string{"", "anthropic/", "openrouter/anthropic/", "deepinfra/anthropic/"} {
			assert.Equal(t, tableAnswers(frame+release), tableAnswers(frame+form), frame+form)
		}
	}
}

func TestAClaudeWordInARoutePrefixChangesNoAnswer(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"claude-opus-4-6", "claude-opus-latest", "claude-opus-6", "claude-5.5-opus", "gpt-5.7", "glm-5.4", "gemini-4-pro",
	} {
		for _, prefix := range []string{
			"claude-proxy/", "@claude-team/", "projects/claude-team/locations/us-central1/publishers/google/models/",
		} {
			neutral, prefixed := strings.ReplaceAll(prefix, "claude-", "relay-")+model, prefix+model
			assert.Equal(t, tableAnswers(neutral), tableAnswers(prefixed), prefixed)
			documented, inherited := InheritedModel(neutral)
			gotDocumented, gotInherited := InheritedModel(prefixed)
			assert.Equal(t, []any{documented, inherited}, []any{gotDocumented, gotInherited}, prefixed)
		}
	}
}

func TestCanonicalClaudeLeavesUnrelatedDotsAlone(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ in, want string }{
		{"claude-opus-4.7", "claude-opus-4-7"},
		{"us.anthropic.claude-opus-4-7-v1:0", "us.anthropic.claude-opus-4-7"},
		{"claude-sonnet-4@20250514", "claude-sonnet-4-0"},
		{"CLAUDE-Opus-4.6", "claude-opus-4-6"},
		{"gpt-4.1", "gpt-4-1"},
	} {
		if got := canonicalClaude(tc.in); got != tc.want {
			t.Errorf("canonicalClaude(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestDetectionSeesPlatformIdsAndDashedVersions(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model string
		want  bool
	}{
		{"us.anthropic.claude-opus-5-v1:0", true},
		{"us.anthropic.claude-sonnet-4-6-v1:0", true},
		{"eu.anthropic.claude-opus-4-6-v1:0", true},
		{"claude-3-7-sonnet-20250219", true},
		{"claude-3.7-sonnet", true},
		{"claude-haiku-4-5", true},
		{"glm-4.5", true},
		{"deepseek-v3.1-terminus", true},
		{"us.amazon.titan-text-lite-v1", false},
	} {
		if got := IsReasoningModel(tc.model); got != tc.want {
			t.Errorf("IsReasoningModel(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestOpenAISuffixVariantsDoNotInheritTheBaseRules(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model   string
		efforts []string
	}{
		{"gpt-5-pro", []string{"high"}},
		{"gpt-5.1-pro", []string{"high"}},
		{"gpt-5.2-pro", []string{"medium", "high", "xhigh"}},
		{"gpt-5.4-pro", []string{"medium", "high", "xhigh"}},
	} {
		caps := OpenAIReasoningCapsFor(tc.model)
		if !caps.Known || caps.CanDisable || !slices.Equal(caps.Efforts, tc.efforts) {
			t.Errorf("%s: want the Pro caps (%v, not disablable), got %+v", tc.model, tc.efforts, caps)
		}
	}

	for _, model := range []string{"gpt-5-chat-latest", "gpt-5.2-chat-latest"} {
		if caps := OpenAIReasoningCapsFor(model); caps.Known {
			t.Errorf("%s: a chat variant must not be classified as a reasoning model, got %+v", model, caps)
		}
		if IsReasoningModel(model) {
			t.Errorf("%s: a chat variant must not be detected as reasoning", model)
		}
	}
}

func TestClaudeFourAliasesAreClassified(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model string
		want  bool
	}{
		{"claude-opus-4-0", true},
		{"claude-sonnet-4-0", true},
		{"claude-sonnet-4@20250514", true},
		{"claude-opus-4@20250514", true},
		{"claude-opus-4-20250514", true},
		{"claude-sonnet-4-5", false},
		{"claude-sonnet-4-6", false},
		{"claude-opus-4-6", false},
	} {
		if got := ClaudePredatesAdaptive(tc.model); got != tc.want {
			t.Errorf("ClaudePredatesAdaptive(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestStructuredOutputGateSeesEverySpellingOfTheSameGeneration(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"claude-opus-4-0", "claude-sonnet-4-0",
		"claude-opus-4@20250514", "claude-sonnet-4@20250514",
		"claude-opus-4-20250514", "claude-sonnet-4-20250514",
		"us.anthropic.claude-sonnet-4-0",
	} {
		if ClaudeSupportsStructuredOutput(model) {
			t.Errorf("%q predates structured output, so it must be refused locally", model)
		}
	}

	for _, model := range []string{
		"claude-sonnet-4-5", "claude-opus-4-5", "claude-opus-4-6", "claude-opus-5",
	} {
		if !ClaudeSupportsStructuredOutput(model) {
			t.Errorf("%q takes structured output", model)
		}
	}
}

func TestDottedSpellingsSurviveTheirTableEntriesBeingDropped(t *testing.T) {
	t.Parallel()

	if got := ClaudeReasoningKindFor("claude-haiku-4.5"); got != ClaudeReasoningBudgetOnly {
		t.Errorf("ClaudeReasoningKindFor(dotted haiku) = %v, want budget-only", got)
	}
	if ClaudeSupportsStructuredOutput("claude-3.5-sonnet") {
		t.Error("the dotted Claude 3 spelling predates structured output too")
	}
}

func TestLikelyReasoningModelIgnoresThePlatformSpelling(t *testing.T) {
	t.Parallel()

	for _, pair := range [][2]string{
		{"claude-9-sonnet", "us.anthropic.claude-9-sonnet"},
		{"gpt-9", "azure.gpt-9"},
	} {
		bare, platform := LikelyReasoningModel(pair[0]), LikelyReasoningModel(pair[1])
		if bare != platform {
			t.Errorf("hint differs by spelling: %q = %v, %q = %v", pair[0], bare, pair[1], platform)
		}
	}
}

func TestClaudePreThinkingStopsBelow37(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-3-7-sonnet-20250219", "claude-3.7-sonnet"} {
		if claudePreThinking(model) {
			t.Errorf("%q is where extended thinking starts, not below it", model)
		}
	}
	for _, model := range []string{"claude-3-5-sonnet", "claude-3.5-sonnet", "claude-2", "claude-instant-1"} {
		if !claudePreThinking(model) {
			t.Errorf("%q predates extended thinking", model)
		}
	}
}

func TestClaudeClampEffortMovesToTheNearestAcceptedLevel(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ model, effort, want string }{
		{"claude-opus-4-7", "xhigh", "xhigh"},
		{"claude-sonnet-5", "xhigh", "xhigh"},
		{"claude-opus-4-6", "xhigh", "high"},
		{"claude-sonnet-4-6", "xhigh", "high"},
		{"claude-opus-4-5", "xhigh", "high"},
		{"claude-opus-4-5", "max", "high"},
		{"claude-haiku-4-5", "max", "high"},
		{"claude-opus-4-6", "max", "max"},
		{"claude-opus-4-6", "low", "low"},
		{"claude-opus-4-6", "minimal", "low"},
		{"claude-sonnet-5", "minimal", "low"},
		{"claude-opus-4-5", "minimal", "low"},
		{"grok-4", "minimal", "minimal"},
		{"grok-4", "xhigh", "xhigh"},
		{"claude-opus-4-6", "", ""},
	} {
		if got := ClaudeClampEffort(tc.model, tc.effort, ProviderAnthropic); got != tc.want {
			t.Errorf("ClaudeClampEffort(%q, %q) = %q, want %q", tc.model, tc.effort, got, tc.want)
		}
	}
}

func TestBedrockServesTheTopEffortsOnlyWhereItsGuideSays(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model string
		want  []string
	}{
		{"us.anthropic.claude-opus-5", []string{"low", "medium", "high", "xhigh", "max"}},
		{"us.anthropic.claude-opus-4-6-v1", []string{"low", "medium", "high", "max"}},
		{"us.anthropic.claude-sonnet-4-6", []string{"low", "medium", "high", "max"}},
		{"us.anthropic.claude-sonnet-5", []string{"low", "medium", "high"}},
		{"us.anthropic.claude-opus-4-7", []string{"low", "medium", "high"}},
		{"us.anthropic.claude-opus-4-8", []string{"low", "medium", "high"}},
		{"us.anthropic.claude-fable-5", []string{"low", "medium", "high"}},
		{"us.anthropic.claude-fable-5-1", []string{"low", "medium", "high"}},
	} {
		if got := ClaudeEffortsFor(tc.model, ProviderBedrock); !slices.Equal(got, tc.want) {
			t.Errorf("ClaudeEffortsFor(%q, Bedrock) = %v, want %v", tc.model, got, tc.want)
		}
	}

	for _, tc := range []struct{ model, effort, want string }{
		{"us.anthropic.claude-sonnet-5", "xhigh", "high"},
		{"us.anthropic.claude-sonnet-5", "max", "high"},
		{"us.anthropic.claude-opus-4-7", "max", "high"},
		{"us.anthropic.claude-fable-5-1", "xhigh", "high"},
		{"us.anthropic.claude-opus-4-6-v1", "xhigh", "high"},
		{"us.anthropic.claude-opus-4-6-v1", "max", "max"},
		{"us.anthropic.claude-opus-5", "xhigh", "xhigh"},
		{"us.anthropic.claude-opus-5", "max", "max"},
	} {
		if got := ClaudeClampEffort(tc.model, tc.effort, ProviderBedrock); got != tc.want {
			t.Errorf("ClaudeClampEffort(%q, %q, Bedrock) = %q, want %q", tc.model, tc.effort, got, tc.want)
		}
	}

	for _, model := range []string{"claude-sonnet-5", "claude-opus-4-7", "claude-fable-5-1"} {
		if got := ClaudeClampEffort(model, "xhigh", ProviderAnthropic); got != "xhigh" {
			t.Errorf("%s on Anthropic keeps xhigh, got %q", model, got)
		}
	}
}
