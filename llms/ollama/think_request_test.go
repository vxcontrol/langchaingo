package ollama

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func captureChatRequest(t *testing.T, opts ...llms.CallOption) map[string]any {
	t.Helper()
	return captureChatRequestFor(t, "glm-5", opts...)
}

func captureChatRequestFor(t *testing.T, model string, opts ...llms.CallOption) map[string]any {
	t.Helper()

	body, err := sendChatRequest(t, model, opts...)
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(body, &got))
	return got
}

func sendChatRequest(t *testing.T, model string, opts ...llms.CallOption) ([]byte, error) {
	t.Helper()

	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"` + model + `","message":{"role":"assistant","content":"ok"},` +
			`"done":true,"done_reason":"stop"}` + "\n"))
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel(model))
	require.NoError(t, err)

	_, err = llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	return body, err
}

func TestThinkReachesTheWire(t *testing.T) {
	t.Parallel()

	t.Run("disabled reasoning is sent, not left to the server", func(t *testing.T) {
		t.Parallel()
		require.Equal(t, false, captureChatRequest(t, llms.WithReasoningDisabled())["think"])
	})

	t.Run("a level ollama accepts travels as itself", func(t *testing.T) {
		t.Parallel()
		require.Equal(t, "low", captureChatRequest(t, llms.WithReasoning(llms.ReasoningLow, 0))["think"])
	})

	t.Run("a level ollama does not accept falls back to plain on", func(t *testing.T) {
		t.Parallel()
		require.Equal(t, true, captureChatRequest(t, llms.WithReasoning(llms.ReasoningXHigh, 0))["think"])
	})

	t.Run("saying nothing leaves the field off the wire", func(t *testing.T) {
		t.Parallel()
		require.NotContains(t, captureChatRequest(t), "think")
	})
}

func TestGPTOSSNeverGetsAThinkValueItIgnores(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"gpt-oss", "gpt-oss:120b", "gpt-oss:20b", "library/gpt-oss:20b"} {
		for _, effort := range []llms.ReasoningEffort{
			llms.ReasoningMinimal, llms.ReasoningLow, llms.ReasoningMedium,
			llms.ReasoningHigh, llms.ReasoningXHigh, llms.ReasoningMax,
		} {
			think := captureChatRequestFor(t, model, llms.WithReasoning(effort, 0))["think"]
			require.Contains(t, []any{"low", "medium", "high"}, think,
				"%s asked %s", model, effort)
		}
	}
}

func TestDisablingThinkingOnGPTOSSIsRefusedBeforeTheRequest(t *testing.T) {
	t.Parallel()

	body, err := sendChatRequest(t, "gpt-oss:120b", llms.WithReasoningDisabled())

	var offErr *reasoning.ErrReasoningOffUnsupported
	require.ErrorAs(t, err, &offErr)
	require.Empty(t, body, "the request must not reach the server")
}

func TestAnEffortGPTOSSCannotTakeIsReported(t *testing.T) {
	t.Parallel()

	for asked, sent := range map[llms.ReasoningEffort]string{
		llms.ReasoningMinimal: "low",
		llms.ReasoningXHigh:   "high",
		llms.ReasoningMax:     "high",
	} {
		resp := generateForWarningsOn(t, "gpt-oss:120b", llms.WithReasoning(asked, 0))

		w, ok := ollamaWarningsByOption(resp.Warnings)["WithReasoning"]
		require.True(t, ok, "no warning for %s in %v", asked, resp.Warnings)
		require.Equal(t, llms.WarningSubstitute, w.Kind)
		require.Equal(t, string(asked), w.Asked)
		require.Equal(t, sent, w.Sent)
	}
}

func TestAnEffortGPTOSSTakesIsNotAWarning(t *testing.T) {
	t.Parallel()

	for _, effort := range []llms.ReasoningEffort{llms.ReasoningLow, llms.ReasoningMedium, llms.ReasoningHigh} {
		resp := generateForWarningsOn(t, "gpt-oss:120b", llms.WithReasoning(effort, 0))
		require.Empty(t, resp.Warnings, "%s is in the vendor set, nothing is lost", effort)
	}
}
