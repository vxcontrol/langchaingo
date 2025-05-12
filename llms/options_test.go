package llms

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReasoningConfig_IsEnabled(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		config   *ReasoningConfig
		expected bool
	}{
		{
			name:     "nil config",
			config:   nil,
			expected: false,
		},
		{
			name:     "empty config",
			config:   &ReasoningConfig{},
			expected: false,
		},
		{
			name:     "config with tokens only",
			config:   &ReasoningConfig{Tokens: 1000},
			expected: true,
		},
		{
			name:     "config with effort only",
			config:   &ReasoningConfig{Effort: ReasoningLow},
			expected: true,
		},
		{
			name:     "config with both tokens and effort",
			config:   &ReasoningConfig{Effort: ReasoningMedium, Tokens: 2000},
			expected: true,
		},
	}

	for idx := range tests {
		tc := tests[idx]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := tc.config.IsEnabled()
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestReasoningConfig_GetEffort(t *testing.T) {
	t.Parallel()

	maxTokens := 10000

	tests := []struct {
		name      string
		config    *ReasoningConfig
		maxTokens int
		expected  ReasoningEffort
	}{
		{
			name:      "nil config",
			config:    nil,
			maxTokens: maxTokens,
			expected:  ReasoningNone,
		},
		{
			name:      "empty config",
			config:    &ReasoningConfig{},
			maxTokens: maxTokens,
			expected:  ReasoningNone,
		},
		{
			name:      "config with explicit effort",
			config:    &ReasoningConfig{Effort: ReasoningHigh},
			maxTokens: maxTokens,
			expected:  ReasoningHigh,
		},
		{
			name:      "config with low tokens",
			config:    &ReasoningConfig{Tokens: maxTokens / 5},
			maxTokens: maxTokens,
			expected:  ReasoningLow,
		},
		{
			name:      "config with medium tokens",
			config:    &ReasoningConfig{Tokens: maxTokens/4 + 100}, // Just above low threshold
			maxTokens: maxTokens,
			expected:  ReasoningMedium,
		},
		{
			name:      "config with high tokens",
			config:    &ReasoningConfig{Tokens: maxTokens / 2},
			maxTokens: maxTokens,
			expected:  ReasoningHigh,
		},
		{
			name:      "precedence - effort over tokens",
			config:    &ReasoningConfig{Effort: ReasoningLow, Tokens: maxTokens / 2},
			maxTokens: maxTokens,
			expected:  ReasoningLow,
		},
		{
			name:      "negative maxTokens uses default 8192",
			config:    &ReasoningConfig{Tokens: 8192/3 + 10}, // Just above medium threshold for 8192
			maxTokens: -1,
			expected:  ReasoningHigh,
		},
		{
			name:      "zero maxTokens uses default 8192",
			config:    &ReasoningConfig{Tokens: 8192 / 5}, // Low for 8192
			maxTokens: 0,
			expected:  ReasoningLow,
		},
	}

	for idx := range tests {
		tc := tests[idx]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := tc.config.GetEffort(tc.maxTokens)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestReasoningConfig_GetTokens(t *testing.T) {
	t.Parallel()

	maxTokens := 12000

	tests := []struct {
		name      string
		config    *ReasoningConfig
		maxTokens int
		expected  int
	}{
		{
			name:      "nil config",
			config:    nil,
			maxTokens: maxTokens,
			expected:  0,
		},
		{
			name:      "empty config",
			config:    &ReasoningConfig{},
			maxTokens: maxTokens,
			expected:  0,
		},
		{
			name:      "config with explicit tokens",
			config:    &ReasoningConfig{Tokens: 3000},
			maxTokens: maxTokens,
			expected:  3000,
		},
		{
			name:      "config with low effort",
			config:    &ReasoningConfig{Effort: ReasoningLow},
			maxTokens: maxTokens,
			expected:  max(maxTokens/4, 1024),
		},
		{
			name:      "config with medium effort",
			config:    &ReasoningConfig{Effort: ReasoningMedium},
			maxTokens: maxTokens,
			expected:  max(maxTokens/3, 2048),
		},
		{
			name:      "config with high effort",
			config:    &ReasoningConfig{Effort: ReasoningHigh},
			maxTokens: maxTokens,
			expected:  max(maxTokens/2, 4096),
		},
		{
			name:      "tokens exceeding max reasoning tokens",
			config:    &ReasoningConfig{Tokens: MaxReasoningTokens + 1000},
			maxTokens: maxTokens,
			expected:  min(MaxReasoningTokens, maxTokens*2/3),
		},
		{
			name:      "tokens exceeding 2/3 of max tokens",
			config:    &ReasoningConfig{Tokens: maxTokens},
			maxTokens: maxTokens,
			expected:  maxTokens * 2 / 3,
		},
		{
			name:      "invalid effort",
			config:    &ReasoningConfig{Effort: "invalid"},
			maxTokens: maxTokens,
			expected:  -1,
		},
		{
			name:      "negative maxTokens uses default 8192 for low effort",
			config:    &ReasoningConfig{Effort: ReasoningLow},
			maxTokens: -10,
			expected:  max(8192/4, 1024), // Based on default 8192
		},
		{
			name:      "zero maxTokens uses default 8192 for high effort",
			config:    &ReasoningConfig{Effort: ReasoningHigh},
			maxTokens: 0,
			expected:  max(8192/2, 4096), // Based on default 8192
		},
		{
			name:      "negative maxTokens with explicit tokens",
			config:    &ReasoningConfig{Tokens: 7000},
			maxTokens: -5,
			expected:  min(7000, 8192*2/3), // Using default 8192 as maxTokens
		},
	}

	for idx := range tests {
		tc := tests[idx]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := tc.config.GetTokens(tc.maxTokens)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestWithReasoning(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		effort          ReasoningEffort
		tokens          int
		expectedEffort  ReasoningEffort
		expectedTokens  int
		expectedEnabled bool
	}{
		{
			name:            "high effort",
			effort:          ReasoningHigh,
			tokens:          0,
			expectedEffort:  ReasoningHigh,
			expectedTokens:  0,
			expectedEnabled: true,
		},
		{
			name:            "medium effort",
			effort:          ReasoningMedium,
			tokens:          0,
			expectedEffort:  ReasoningMedium,
			expectedTokens:  0,
			expectedEnabled: true,
		},
		{
			name:            "low effort",
			effort:          ReasoningLow,
			tokens:          0,
			expectedEffort:  ReasoningLow,
			expectedTokens:  0,
			expectedEnabled: true,
		},
		{
			name:            "specific tokens",
			effort:          ReasoningNone,
			tokens:          5000,
			expectedEffort:  ReasoningNone,
			expectedTokens:  5000,
			expectedEnabled: true,
		},
		{
			name:            "both effort and tokens",
			effort:          ReasoningHigh,
			tokens:          3000,
			expectedEffort:  ReasoningHigh,
			expectedTokens:  3000,
			expectedEnabled: true,
		},
		{
			name:            "disabled reasoning",
			effort:          ReasoningNone,
			tokens:          0,
			expectedEffort:  ReasoningNone,
			expectedTokens:  0,
			expectedEnabled: false,
		},
	}

	for idx := range tests {
		tc := tests[idx]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			opts := &CallOptions{}
			WithReasoning(tc.effort, tc.tokens)(opts)

			assert.NotNil(t, opts.Reasoning)
			assert.Equal(t, tc.expectedEffort, opts.Reasoning.Effort)
			assert.Equal(t, tc.expectedTokens, opts.Reasoning.Tokens)
			assert.Equal(t, tc.expectedEnabled, opts.Reasoning.IsEnabled())
		})
	}
}

func TestReasoningConfig_Integration(t *testing.T) {
	t.Parallel()

	const maxTokens = 16000

	tests := []struct {
		name            string
		config          *ReasoningConfig
		maxTokens       int
		expectedEnabled bool
		expectedEffort  ReasoningEffort
		expectedTokens  int
	}{
		{
			name:            "high effort conversion to tokens",
			config:          &ReasoningConfig{Effort: ReasoningHigh},
			maxTokens:       maxTokens,
			expectedEnabled: true,
			expectedEffort:  ReasoningHigh,
			expectedTokens:  8000, // max(maxTokens/2, 4096)
		},
		{
			name:            "tokens conversion to effort level",
			config:          &ReasoningConfig{Tokens: 2500},
			maxTokens:       maxTokens,
			expectedEnabled: true,
			expectedEffort:  ReasoningLow, // tokens < maxTokens/4 (4000)
			expectedTokens:  2500,
		},
		{
			name:            "cap excessive tokens",
			config:          &ReasoningConfig{Tokens: maxTokens},
			maxTokens:       maxTokens,
			expectedEnabled: true,
			expectedEffort:  ReasoningHigh, // tokens > maxTokens/3
			expectedTokens:  maxTokens * 2 / 3,
		},
		{
			name:            "default maxTokens handling",
			config:          &ReasoningConfig{Effort: ReasoningMedium},
			maxTokens:       0, // Should use default 8192
			expectedEnabled: true,
			expectedEffort:  ReasoningMedium,
			expectedTokens:  max(8192/3, 2048), // Based on default 8192
		},
	}

	for idx := range tests {
		tc := tests[idx]
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			isEnabled := tc.config.IsEnabled()
			effort := tc.config.GetEffort(tc.maxTokens)
			tokens := tc.config.GetTokens(tc.maxTokens)

			assert.Equal(t, tc.expectedEnabled, isEnabled)
			assert.Equal(t, tc.expectedEffort, effort)
			assert.Equal(t, tc.expectedTokens, tokens)
		})
	}
}
