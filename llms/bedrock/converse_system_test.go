package bedrock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestConverseSendsEverySystemPartInOrder(t *testing.T) {
	t.Parallel()

	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, converseAnswer)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv,
		bedrock.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())
	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "S1"),
		llms.TextParts(llms.ChatMessageTypeSystem, "S2", "S3"),
		llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
	})
	require.NoError(t, err)

	var sent struct {
		System []map[string]any `json:"system"`
	}
	require.NoError(t, json.Unmarshal(body, &sent))
	require.Equal(t, []map[string]any{{"text": "S1"}, {"text": "S2"}, {"text": "S3"}}, sent.System)
}
