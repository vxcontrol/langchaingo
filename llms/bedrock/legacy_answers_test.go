package bedrock_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func TestTheLegacyDoorAnswersWithEveryCandidateTheFamilyReturns(t *testing.T) {
	t.Parallel()

	for _, family := range []struct{ model, answer string }{
		{"ai21.j2-ultra-v1", `{"id":1,"prompt":{"tokens":[{},{},{}]},"completions":[` +
			`{"data":{"text":"one","tokens":[{},{}]},"finishReason":{"reason":"endoftext"}},` +
			`{"data":{"text":"two","tokens":[{}]},"finishReason":{"reason":"length"}}]}`},
		{"ai21.jamba-1-5-large-v1:0", `{"id":"x","choices":[` +
			`{"index":0,"message":{"role":"assistant","content":"one"},"finish_reason":"stop"},` +
			`{"index":1,"message":{"role":"assistant","content":"two"},"finish_reason":"length"}],` +
			`"usage":{"prompt_tokens":3,"completion_tokens":3,"total_tokens":6}}`},
		{"cohere.command-text-v14", `{"generations":[{"text":"one","finish_reason":"COMPLETE"},` +
			`{"text":"two","finish_reason":"MAX_TOKENS"}]}`},
	} {
		llm, _ := legacyLLMCapturing(t, family.answer, bedrock.WithModel(family.model))
		resp, err := llm.GenerateContent(t.Context(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithCandidateCount(2))
		require.NoError(t, err, family.model)
		require.Len(t, resp.Choices, 2, family.model)
		require.Equal(t, "one", resp.Choices[0].Content, family.model)
		require.Equal(t, "two", resp.Choices[1].Content, family.model)
		require.True(t, resp.Choices[1].Truncated, "%s: the second candidate hit the limit", family.model)
	}

	llm, _ := legacyLLMCapturing(t, `{"id":1,"prompt":{"tokens":[{},{},{}]},"completions":[`+
		`{"data":{"text":"one","tokens":[{},{}]},"finishReason":{"reason":"endoftext"}}]}`,
		bedrock.WithModel("ai21.j2-mid-v1"))
	resp, err := llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	require.NoError(t, err)
	info := resp.Choices[0].GenerationInfo
	require.Equal(t, 3, info["PromptTokens"], "Jurassic-2 counts the prompt's tokens")
	require.Equal(t, 2, info["CompletionTokens"])
	require.Equal(t, 5, info["TotalTokens"])
}

func TestAnUnreadableLegacyAnswerIsAnError(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"ai21.j2-ultra-v1", "meta.llama3-70b-instruct-v1:0", "amazon.titan-text-express-v1"} {
		llm, _ := legacyLLMCapturing(t, "not json", bedrock.WithModel(model))
		_, err := llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
		require.Error(t, err, model)
	}
}

func TestABrokenLegacyStreamChunkEndsTheStreamWithAnErrorAndTheTextSoFar(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		enc := eventstream.NewEncoder()
		writeLegacyChunk(t, w, enc, `{"type":"message_start","message":{"id":"x","type":"message",`+
			`"role":"assistant","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}`)
		writeLegacyChunk(t, w, enc, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"first "}}`)
		writeLegacyChunk(t, w, enc, `{"type":"content_block_delta","index":0,"delta":`)
		writeLegacyChunk(t, w, enc, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"second"}}`)
		writeLegacyChunk(t, w, enc, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv, bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"))
	resp, err := llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
	require.Error(t, err)
	require.NotNil(t, resp, "the text that arrived is kept")
	require.Equal(t, "first ", resp.Choices[0].Content)
}
