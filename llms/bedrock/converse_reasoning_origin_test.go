package bedrock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func converseReplayedReasoning(t *testing.T, model string, thought *reasoning.ContentReasoning) []any {
	t.Helper()

	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, converseAnswer)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv, bedrock.WithModel(model), bedrock.WithConverseAPI())
	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("earlier answer", thought)}},
		llms.TextParts(llms.ChatMessageTypeHuman, "go on"),
	})
	require.NoError(t, err)

	var sent struct {
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(body, &sent))
	var thinking []any
	for _, block := range sent.Messages[1].Content {
		if block["reasoningContent"] != nil {
			thinking = append(thinking, block["reasoningContent"])
		}
	}
	return thinking
}

func TestClaudeOnConverseGetsBackOnlyTheThinkingItCanVerify(t *testing.T) {
	t.Parallel()

	signedBy := func(model string) *reasoning.ContentReasoning {
		return (&reasoning.ContentReasoning{Content: "plan", Signature: []byte("sig")}).WrittenBy(model)
	}
	kept := []any{map[string]any{"reasoningText": map[string]any{"text": "plan", "signature": "sig"}}}

	require.Equal(t, kept, converseReplayedReasoning(t, converseClaude, signedBy("claude-opus-4-8")))
	require.Equal(t, kept, converseReplayedReasoning(t, converseClaude, signedBy("")))
	require.Empty(t, converseReplayedReasoning(t, converseClaude, signedBy("gemini-2.5-pro")))
	require.Empty(t, converseReplayedReasoning(t, converseClaude,
		(&reasoning.ContentReasoning{Content: "plan"}).WrittenBy("deepseek-reasoner")))
	require.Empty(t, converseReplayedReasoning(t, converseClaude, &reasoning.ContentReasoning{Content: "plan"}))
}

func TestAConverseModelThatIsNotClaudeGetsItsReasoningAsBefore(t *testing.T) {
	t.Parallel()

	got := converseReplayedReasoning(t, "us.deepseek.r1-v1:0",
		(&reasoning.ContentReasoning{Content: "plan"}).WrittenBy("us.deepseek.r1-v1:0"))
	require.Equal(t, []any{map[string]any{"reasoningText": map[string]any{"text": "plan"}}}, got)
}

func TestAConverseAnswersReasoningNamesTheModelThatWroteIt(t *testing.T) {
	t.Parallel()

	ask := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}

	whole := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"output":{"message":{"role":"assistant","content":[`+
			`{"reasoningContent":{"reasoningText":{"text":"plan","signature":"sig"}}},{"text":"ok"}]}},`+
			`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
	}))
	t.Cleanup(whole.Close)
	resp, err := bedrockLLMAgainst(t, whole, bedrock.WithModel(converseClaude), bedrock.WithConverseAPI()).
		GenerateContent(context.Background(), ask)
	require.NoError(t, err)
	require.Equal(t, converseClaude, resp.Choices[0].Reasoning.Model)

	streamed := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		enc := eventstream.NewEncoder()
		writeConverseEvent(t, w, enc, "messageStart", `{"role":"assistant"}`)
		writeConverseEvent(t, w, enc, "contentBlockDelta",
			`{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"plan"}}}`)
		writeConverseEvent(t, w, enc, "contentBlockDelta",
			`{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"sig"}}}`)
		writeConverseEvent(t, w, enc, "contentBlockStop", `{"contentBlockIndex":0}`)
		writeConverseEvent(t, w, enc, "messageStop", `{"stopReason":"end_turn"}`)
		writeConverseEvent(t, w, enc, "metadata", `{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
	}))
	t.Cleanup(streamed.Close)
	resp, err = bedrockLLMAgainst(t, streamed, bedrock.WithModel(converseClaude), bedrock.WithConverseAPI()).
		GenerateContent(context.Background(), ask,
			llms.WithStreamingFunc(func(_ context.Context, _ streaming.Chunk) error { return nil }))
	require.NoError(t, err)
	require.Equal(t, converseClaude, resp.Choices[0].Reasoning.Model)
}
