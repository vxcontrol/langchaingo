package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func gatewayAnswering(t *testing.T, contentType, answer string, opts ...Option) (*LLM, *[]byte) {
	t.Helper()

	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", contentType)
		_, _ = io.WriteString(w, answer)
	}))
	t.Cleanup(srv.Close)

	return newUnitLLM(t, append([]Option{WithBaseURL(srv.URL)}, opts...)...), &raw
}

func TestAGatewaysThinkingBlocksKeepTheirSignature(t *testing.T) {
	t.Parallel()

	want := []reasoning.Block{{Text: "plan", Signature: []byte("sig1")}, {Redacted: []byte("opaque")}}
	ask := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}

	t.Run("in a whole answer", func(t *testing.T) {
		t.Parallel()

		llm, _ := gatewayAnswering(t, "application/json", `{"id":"x","object":"chat.completion","created":1,`+
			`"model":"anthropic/claude-sonnet-4-5","choices":[{"index":0,"message":{"role":"assistant","content":"ok",`+
			`"reasoning_content":"plan","thinking_blocks":[{"type":"thinking","thinking":"plan","signature":"sig1"},`+
			`{"type":"redacted_thinking","data":"opaque"}]},"finish_reason":"stop"}]}`,
			WithModel("anthropic/claude-sonnet-4-5"))
		resp, err := llm.GenerateContent(context.Background(), ask)
		require.NoError(t, err)
		require.Equal(t, want, resp.Choices[0].Reasoning.Sequence())
	})

	t.Run("in a stream", func(t *testing.T) {
		t.Parallel()

		chunk := func(delta string) string {
			return `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` +
				delta + `}]}` + "\n\n"
		}
		llm, _ := gatewayAnswering(t, "text/event-stream",
			chunk(`{"role":"assistant","reasoning_content":"pl","thinking_blocks":[{"type":"thinking","thinking":"pl"}]}`)+
				chunk(`{"reasoning_content":"an","thinking_blocks":[{"type":"thinking","thinking":"an"}]}`)+
				chunk(`{"thinking_blocks":[{"type":"thinking","thinking":"","signature":"sig1"}]}`)+
				chunk(`{"thinking_blocks":[{"type":"redacted_thinking","data":"opaque"}]}`)+
				`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,`+
				`"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n"+
				"data: [DONE]\n\n",
			WithModel("anthropic/claude-sonnet-4-5"))
		resp, err := llm.GenerateContent(context.Background(), ask,
			llms.WithStreamingFunc(func(_ context.Context, _ streaming.Chunk) error { return nil }))
		require.NoError(t, err)
		require.Equal(t, want, resp.Choices[0].Reasoning.Sequence())
	})
}

func TestThinkingBlocksGoBackToAClaudeGatewayOnlyWhenAsked(t *testing.T) {
	t.Parallel()

	signed := reasoning.FromBlocks([]reasoning.Block{{Text: "plan", Signature: []byte("sig1")}})
	unsigned := &reasoning.ContentReasoning{Content: "plan"}
	sent := func(t *testing.T, model string, thought *reasoning.ContentReasoning, opts ...Option) map[string]any {
		t.Helper()

		llm, raw := gatewayAnswering(t, "application/json", `{"id":"x","object":"chat.completion","created":1,`+
			`"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
			append([]Option{WithModel(model)}, opts...)...)
		_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "look it up"),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.TextPartWithReasoning("", thought),
				llms.ToolCall{ID: "c1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: `{}`}},
			}},
			{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
				llms.ToolCallResponse{ToolCallID: "c1", Name: "lookup", Content: "done"},
			}},
		})
		require.NoError(t, err)

		var body struct {
			Messages []map[string]any `json:"messages"`
		}
		require.NoError(t, json.Unmarshal(*raw, &body))
		return body.Messages[1]
	}

	t.Run("signed blocks go back when replay is on", func(t *testing.T) {
		t.Parallel()
		got := sent(t, "anthropic/claude-sonnet-4-5", signed, WithPreserveReasoningContent())
		require.Equal(t, []any{map[string]any{"type": "thinking", "thinking": "plan", "signature": "sig1"}},
			got["thinking_blocks"])
	})
	t.Run("nothing goes back by default", func(t *testing.T) {
		t.Parallel()
		require.NotContains(t, sent(t, "anthropic/claude-sonnet-4-5", signed), "thinking_blocks")
	})
	t.Run("an unsigned thought is not a block", func(t *testing.T) {
		t.Parallel()
		require.NotContains(t, sent(t, "anthropic/claude-sonnet-4-5", unsigned, WithPreserveReasoningContent()),
			"thinking_blocks")
	})
	t.Run("a model that is not Claude takes no blocks", func(t *testing.T) {
		t.Parallel()
		require.NotContains(t, sent(t, "deepseek/deepseek-chat", signed, WithPreserveReasoningContent()),
			"thinking_blocks")
	})
}
