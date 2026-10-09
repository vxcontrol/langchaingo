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

func replayedThinking(t *testing.T, target string, thought *reasoning.ContentReasoning) []any {
	t.Helper()

	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"m","type":"message","role":"assistant","model":"`+target+`",`+
			`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel(target))
	require.NoError(t, err)
	_, err = llm.GenerateContent(t.Context(), []llms.MessageContent{
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
		if block["type"] == "thinking" || block["type"] == "redacted_thinking" {
			thinking = append(thinking, block)
		}
	}
	return thinking
}

func TestClaudeGetsBackOnlyTheThinkingItCanVerify(t *testing.T) {
	t.Parallel()

	const target = "claude-sonnet-4-5"
	signedBy := func(model string) *reasoning.ContentReasoning {
		return (&reasoning.ContentReasoning{Content: "plan", Signature: []byte("sig")}).WrittenBy(model)
	}
	kept := []any{map[string]any{"type": "thinking", "thinking": "plan", "signature": "sig"}}

	require.Equal(t, kept, replayedThinking(t, target, signedBy("claude-opus-4-8")), "another Claude model wrote it")
	require.Equal(t, kept, replayedThinking(t, target, signedBy("")), "written before the writer was recorded")
	require.Empty(t, replayedThinking(t, target, signedBy("gemini-2.5-pro")), "a Gemini signature")
	require.Empty(t, replayedThinking(t, target, (&reasoning.ContentReasoning{Content: "plan"}).WrittenBy("deepseek-reasoner")),
		"thinking without a signature")
	require.Empty(t, replayedThinking(t, target, &reasoning.ContentReasoning{Content: "plan"}),
		"thinking without a signature and without a writer")
}

func TestAModelThatIsNotClaudeGetsItsThinkingAsBefore(t *testing.T) {
	t.Parallel()

	got := replayedThinking(t, "deepseek-chat", (&reasoning.ContentReasoning{Content: "plan"}).WrittenBy("deepseek-chat"))
	require.Equal(t, []any{map[string]any{"type": "thinking", "thinking": "plan"}}, got)
}

func TestAnAnswersThinkingNamesTheModelThatWroteIt(t *testing.T) {
	t.Parallel()

	llm := answeringLLM(t, "application/json", `{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-4-5",`+
		`"content":[{"type":"thinking","thinking":"plan","signature":"sig"},{"type":"text","text":"ok"}],`+
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	resp, err := llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	require.NoError(t, err)
	require.Equal(t, "claude-sonnet-4-5", resp.Choices[0].Reasoning.Model)
}
