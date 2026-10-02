package reasoning

import (
	"maps"
	"slices"
	"strings"
	"testing"

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
		}
	}
}
