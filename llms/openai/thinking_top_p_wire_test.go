package openai

import (
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestClaudeThinkingKeepsTopPAboveTheFloor(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name   string
		model  string
		opts   []llms.CallOption
		want   string
		absent bool
	}{
		{"Claude keeps top_p above the floor", "claude-sonnet-4-5",
			[]llms.CallOption{llms.WithTopP(0.97)}, `"top_p":0.97`, false},
		{"Claude keeps top_p exactly at the floor", "claude-sonnet-4-5",
			[]llms.CallOption{llms.WithTopP(0.95)}, `"top_p":0.95`, false},
		{"Claude strips top_p below the floor", "claude-sonnet-4-5",
			[]llms.CallOption{llms.WithTopP(0.5)}, `"top_p"`, true},
		{"caller set both — top_p is dropped", "claude-sonnet-4-5",
			[]llms.CallOption{llms.WithTopP(0.97), llms.WithTemperature(0.3)}, `"top_p"`, true},
		{"model without sampling does not get top_p", "claude-sonnet-5",
			[]llms.CallOption{llms.WithTopP(0.97)}, `"top_p"`, true},
		{"OpenAI reasoning family still drops top_p", "gpt-5.4",
			[]llms.CallOption{llms.WithTopP(0.97)}, `"top_p"`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			opts := append([]llms.CallOption{
				llms.WithMaxTokens(8000), llms.WithReasoning(llms.ReasoningHigh, 0),
			}, tc.opts...)
			body := bodyForCall(t, tc.model, opts...)
			if got := strings.Contains(body, tc.want); got == tc.absent {
				verb := "does not contain"
				if tc.absent {
					verb = "contains"
				}
				t.Errorf("request body %s %s\nbody: %s", verb, tc.want, body)
			}
		})
	}
}
