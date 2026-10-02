package anthropic_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestAnUnlistedClaudeVersionIsSentTheShapeOfTheReleaseItFollows(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-opus-6", "claude-opus-5-6", "claude-opus-4-10"} {
		body, _ := captureMessagesRequestModel(t, model,
			llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithTemperature(0.7))
		thinking, _ := body["thinking"].(map[string]any)
		require.Equal(t, "adaptive", thinking["type"], "%s: %v", model, body)
		require.NotContains(t, thinking, "budget_tokens", model)
		require.NotContains(t, body, "temperature", model)
		config, _ := body["output_config"].(map[string]any)
		require.Equal(t, "high", config["effort"], "%s: %v", model, body)
	}
}

func generateRecording(t *testing.T, model string, opts ...llms.CallOption) (map[string]any, *llms.ContentResponse, error) {
	t.Helper()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg","type":"message","role":"assistant","model":"m",` +
			`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel(model))
	require.NoError(t, err)
	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	return body, resp, err
}

func inheritWarnings(resp *llms.ContentResponse) map[string]llms.Warning {
	byOption := map[string]llms.Warning{}
	if resp == nil {
		return byOption
	}
	for _, w := range resp.Warnings {
		if w.Kind == llms.WarningInherit {
			byOption[w.Option] = w
		}
	}
	return byOption
}

func TestAnUnlistedClaudeVersionIsSentWhatItsReleaseWouldRefuse(t *testing.T) {
	t.Parallel()

	body, resp, err := generateRecording(t, "claude-opus-6", llms.WithReasoningDisabled())
	require.NoError(t, err)
	thinking, _ := body["thinking"].(map[string]any)
	require.Equal(t, "disabled", thinking["type"], "%v", body)
	warnings := inheritWarnings(resp)
	require.Contains(t, warnings, "WithModel")
	require.Equal(t, "off", warnings["WithReasoningDisabled"].Sent)

	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}
	body, resp, err = generateRecording(t, "claude-sonnet-6",
		llms.WithTools([]llms.Tool{tool}), llms.WithToolChoice(llms.ToolChoice{
			Type: "function", Function: &llms.FunctionReference{Name: "lookup"},
		}))
	require.NoError(t, err)
	choice, _ := body["tool_choice"].(map[string]any)
	require.Equal(t, "lookup", choice["name"], "%v", body)
	require.Equal(t, "lookup", inheritWarnings(resp)["WithToolChoice"].Sent)

	_, _, err = generateRecording(t, "claude-opus-5-5", llms.WithReasoningDisabled())
	var refusal *reasoning.ErrReasoningOffUnsupported
	require.ErrorAs(t, err, &refusal, "the documented release keeps its refusal")
}

func TestAnUnlistedClaudeVersionIsSentTheDisableAndEffortItsLineDocuments(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"claude-fable-6", "claude-mythos-6"} {
		body, resp, err := generateRecording(t, model, llms.WithReasoningDisabled())
		require.NoError(t, err)
		require.NotContains(t, body, "thinking", "no %s release documents a disable: %v", model, body)
		warning, recorded := inheritWarnings(resp)["WithReasoningDisabled"]
		require.True(t, recorded, model)
		require.Empty(t, warning.Sent, model)
	}

	body, resp, err := generateRecording(t, "claude-haiku-5", llms.WithReasoning(llms.ReasoningMinimal, 0))
	require.NoError(t, err)
	config, _ := body["output_config"].(map[string]any)
	require.Equal(t, "low", config["effort"], "%v", body)
	require.Equal(t, "low", inheritWarnings(resp)["WithReasoning"].Sent)
}
