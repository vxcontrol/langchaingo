package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
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

func wholeGatewayAnswer(message string) string {
	return `{"id":"x","object":"chat.completion","created":1,"model":"anthropic/claude-sonnet-4-5",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"ok",` + message + `},"finish_reason":"stop"}]}`
}

func streamedGatewayAnswer(deltas ...string) string {
	var out string
	for _, delta := range deltas {
		out += `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` +
			delta + `}]}` + "\n\n"
	}
	return out + `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,` +
		`"delta":{"content":"ok"},"finish_reason":"stop"}]}` + "\n\n" + "data: [DONE]\n\n"
}

func TestAGatewaysThinkingBlocksKeepTheirSignature(t *testing.T) {
	t.Parallel()

	ask := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
	streamed := llms.WithStreamingFunc(func(_ context.Context, _ streaming.Chunk) error { return nil })
	reasoningOf := func(t *testing.T, contentType, answer string, opts []Option, call ...llms.CallOption) *reasoning.ContentReasoning {
		t.Helper()

		llm, _ := gatewayAnswering(t, contentType, answer, append([]Option{WithModel("anthropic/claude-sonnet-4-5")}, opts...)...)
		resp, err := llm.GenerateContent(context.Background(), ask, call...)
		require.NoError(t, err)
		return resp.Choices[0].Reasoning
	}
	thinkingBlocks := []Option{WithThinkingBlocks()}
	signedAndRedacted := []reasoning.Block{{Text: "plan", Signature: []byte("sig1")}, {Redacted: []byte("opaque")}}

	t.Run("in a whole answer", func(t *testing.T) {
		t.Parallel()
		got := reasoningOf(t, "application/json", wholeGatewayAnswer(`"reasoning_content":"plan","thinking_blocks":[`+
			`{"type":"thinking","thinking":"plan","signature":"sig1"},{"type":"redacted_thinking","data":"opaque"}]`),
			thinkingBlocks)
		require.Equal(t, signedAndRedacted, got.Sequence())
	})

	t.Run("in a stream", func(t *testing.T) {
		t.Parallel()
		got := reasoningOf(t, "text/event-stream", streamedGatewayAnswer(
			`{"role":"assistant","reasoning_content":"pl","thinking_blocks":[{"type":"thinking","thinking":"pl"}]}`,
			`{"reasoning_content":"an","thinking_blocks":[{"type":"thinking","thinking":"an"}]}`,
			`{"thinking_blocks":[{"type":"thinking","thinking":"","signature":"sig1"}]}`,
			`{"thinking_blocks":[{"type":"redacted_thinking","data":"opaque"}]}`,
		), thinkingBlocks, streamed)
		require.Equal(t, signedAndRedacted, got.Sequence())
	})

	t.Run("a block whose thinking is omitted keeps its signature", func(t *testing.T) {
		t.Parallel()
		whole := reasoningOf(t, "application/json", wholeGatewayAnswer(
			`"thinking_blocks":[{"type":"thinking","thinking":"","signature":"sig1"}]`), thinkingBlocks)
		stream := reasoningOf(t, "text/event-stream", streamedGatewayAnswer(
			`{"role":"assistant","thinking_blocks":[{"type":"thinking","thinking":""}]}`,
			`{"thinking_blocks":[{"type":"thinking","thinking":"","signature":"sig1"}]}`,
		), thinkingBlocks, streamed)
		for _, got := range []*reasoning.ContentReasoning{whole, stream} {
			require.Equal(t, []reasoning.Block{{Signature: []byte("sig1")}}, got.Sequence())
		}
	})

	for name, opts := range map[string][]Option{
		"by default":                     nil,
		"when only preserving reasoning": {WithPreserveReasoningContent()},
	} {
		t.Run(name+" the reasoning is what reasoning_content carried", func(t *testing.T) {
			t.Parallel()
			got := reasoningOf(t, "application/json", wholeGatewayAnswer(`"reasoning_content":"plan","thinking_blocks":[`+
				`{"type":"thinking","thinking":"plan","signature":"sig1"}]`), opts)
			require.Equal(t, &reasoning.ContentReasoning{Content: "plan"}, got)
			require.Nil(t, reasoningOf(t, "application/json", wholeGatewayAnswer(
				`"thinking_blocks":[{"type":"thinking","thinking":"","signature":"sig1"}]`), opts))
		})
	}
}

func TestThinkingBlocksGoBackToAClaudeGatewayOnlyWhenAsked(t *testing.T) {
	t.Parallel()

	signed := reasoning.FromBlocks([]reasoning.Block{{Text: "plan", Signature: []byte("sig1")}})
	sentTurns := func(t *testing.T, model string, thought *reasoning.ContentReasoning, opts ...Option) []map[string]any {
		t.Helper()

		llm, raw := gatewayAnswering(t, "application/json", wholeGatewayAnswer(`"reasoning_content":""`),
			append([]Option{WithModel(model)}, opts...)...)
		_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPartWithReasoning("look it up", thought)}},
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
		return body.Messages
	}
	sent := func(t *testing.T, model string, thought *reasoning.ContentReasoning, opts ...Option) map[string]any {
		t.Helper()
		return sentTurns(t, model, thought, opts...)[1]
	}
	const claude = "anthropic/claude-sonnet-4-5"

	t.Run("signed blocks go back when asked", func(t *testing.T) {
		t.Parallel()
		require.Equal(t, []any{map[string]any{"type": "thinking", "thinking": "plan", "signature": "sig1"}},
			sent(t, claude, signed, WithThinkingBlocks())["thinking_blocks"])
	})
	t.Run("a block whose thinking is omitted goes back with an empty thinking", func(t *testing.T) {
		t.Parallel()
		omitted := reasoning.FromBlocks([]reasoning.Block{{Signature: []byte("sig1")}})
		require.Equal(t, []any{map[string]any{"type": "thinking", "thinking": "", "signature": "sig1"}},
			sent(t, claude, omitted, WithThinkingBlocks())["thinking_blocks"])
	})
	t.Run("a redacted block goes back as its data", func(t *testing.T) {
		t.Parallel()
		redacted := reasoning.FromBlocks([]reasoning.Block{{Redacted: []byte("opaque")}})
		require.Equal(t, []any{map[string]any{"type": "redacted_thinking", "data": "opaque"}},
			sent(t, claude, redacted, WithThinkingBlocks())["thinking_blocks"])
	})
	t.Run("nothing goes back by default", func(t *testing.T) {
		t.Parallel()
		require.NotContains(t, sent(t, claude, signed), "thinking_blocks")
	})
	t.Run("preserving reasoning alone sends no blocks", func(t *testing.T) {
		t.Parallel()
		require.NotContains(t, sent(t, claude, signed, WithPreserveReasoningContent()), "thinking_blocks")
	})
	t.Run("an unsigned thought is not a block", func(t *testing.T) {
		t.Parallel()
		require.NotContains(t, sent(t, claude, &reasoning.ContentReasoning{Content: "plan"}, WithThinkingBlocks()),
			"thinking_blocks")
	})
	t.Run("a model that is not Claude takes no blocks", func(t *testing.T) {
		t.Parallel()
		for _, model := range []string{"deepseek/deepseek-chat", "gpt-5", "vertex_ai/gemini-2.5-pro"} {
			require.NotContains(t, sent(t, model, signed, WithThinkingBlocks()), "thinking_blocks", model)
		}
	})
	t.Run("a user turn takes no blocks", func(t *testing.T) {
		t.Parallel()
		require.NotContains(t, sentTurns(t, claude, signed, WithThinkingBlocks())[0], "thinking_blocks")
	})
	t.Run("a public provider host takes no blocks", func(t *testing.T) {
		t.Parallel()
		var raw []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			raw, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, wholeGatewayAnswer(`"reasoning_content":""`))
		}))
		t.Cleanup(srv.Close)
		target, err := url.Parse(srv.URL)
		require.NoError(t, err)

		llm := newUnitLLM(t, WithBaseURL("https://openrouter.ai/api/v1"), WithModel(claude),
			WithThinkingBlocks(), WithHTTPClient(redirectTo{target}))
		_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "look it up"),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("ok", signed)}},
			llms.TextParts(llms.ChatMessageTypeHuman, "go on"),
		})
		require.NoError(t, err)
		require.NotEmpty(t, raw)
		require.NotContains(t, string(raw), "thinking_blocks")
	})
	t.Run("an OpenRouter route takes no blocks", func(t *testing.T) {
		t.Parallel()
		require.NotContains(t, sent(t, "openrouter/anthropic/claude-sonnet-4-5", signed, WithThinkingBlocks()),
			"thinking_blocks")
	})
}

type redirectTo struct{ target *url.URL }

func (r redirectTo) Do(req *http.Request) (*http.Response, error) {
	req.URL.Scheme, req.URL.Host = r.target.Scheme, r.target.Host
	return http.DefaultClient.Do(req)
}
