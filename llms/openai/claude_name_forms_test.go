package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

type sentCall struct {
	body     map[string]any
	warnings []llms.Warning
	err      string
}

func sentFor(t *testing.T, baseURL, model string, messages []llms.MessageContent, opts ...llms.CallOption) sentCall {
	t.Helper()

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
	resp, err := llm.GenerateContent(context.Background(), messages, opts...)
	var sent sentCall
	if err != nil {
		sent.err = fmt.Sprintf("%T", err)
	}
	if doer.body != nil {
		require.NoError(t, json.Unmarshal(doer.body, &sent.body))
		delete(sent.body, "model")
	}
	if resp != nil {
		named := strings.NewReplacer(model, "<model>")
		for _, w := range resp.Warnings {
			w.Model, w.Asked, w.Sent, w.Reason = "", named.Replace(w.Asked), named.Replace(w.Sent), named.Replace(w.Reason)
			sent.warnings = append(sent.warnings, w)
		}
	}
	return sent
}

func TestAClaudeNameFormIsSentWhatItsReleaseIsSent(t *testing.T) {
	t.Parallel()

	human := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
	prefilled := []llms.MessageContent{human[0], llms.TextParts(llms.ChatMessageTypeAI, "Sure,")}
	tools := llms.WithTools([]llms.Tool{astraTool()})
	calls := map[string]struct {
		messages []llms.MessageContent
		opts     []llms.CallOption
	}{
		"sampling":           {human, []llms.CallOption{llms.WithTemperature(0.5), llms.WithTopP(0.9)}},
		"disable":            {human, []llms.CallOption{llms.WithReasoningDisabled()}},
		"forced tool":        {human, []llms.CallOption{tools, llms.WithToolChoice("required")}},
		"effort with tools":  {human, []llms.CallOption{tools, llms.WithReasoning(llms.ReasoningHigh, 0)}},
		"delegated adaptive": {human, []llms.CallOption{llms.WithAdaptiveReasoning(llms.ReasoningNone)}},
		"prefill":            {prefilled, nil},
	}
	for _, pair := range []struct{ baseURL, form, release string }{
		{"https://openrouter.ai/api/v1", "anthropic/claude-5.5-opus", "anthropic/claude-opus-5.5"},
		{"https://openrouter.ai/api/v1", "anthropic/claude-5-fable", "anthropic/claude-fable-5"},
		{"https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4", "anthropic/claude-sonnet-4-20250514"},
		{"http://litellm.internal/v1", "claude-proxy/claude-opus-4-6", "proxy/claude-opus-4-6"},
		{"http://litellm.internal/v1", "claude-proxy/claude-opus-latest", "proxy/claude-opus-latest"},
		{"http://litellm.internal/v1", "claude-relay/gpt-5.7", "relay/gpt-5.7"},
	} {
		for name, call := range calls {
			want := sentFor(t, pair.baseURL, pair.release, call.messages, call.opts...)
			got := sentFor(t, pair.baseURL, pair.form, call.messages, call.opts...)
			assert.Equal(t, want, got, "%s, %s, as %s", pair.form, name, pair.release)
		}
	}
}
