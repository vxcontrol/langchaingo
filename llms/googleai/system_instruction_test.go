package googleai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func recordingServer(t *testing.T, answer string) (*httptest.Server, *[]byte) {
	t.Helper()

	var body []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, answer)
	}))
	t.Cleanup(server.Close)
	return server, &body
}

func systemInstructionParts(t *testing.T, body []byte) []any {
	t.Helper()

	var sent struct {
		SystemInstruction struct {
			Parts []any `json:"parts"`
		} `json:"systemInstruction"`
	}
	require.NoError(t, json.Unmarshal(body, &sent))
	return sent.SystemInstruction.Parts
}

var everySystemMessage = []llms.MessageContent{
	llms.TextParts(llms.ChatMessageTypeSystem, "S1"),
	llms.TextParts(llms.ChatMessageTypeSystem, "S2", "S3"),
	llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
}

var everySystemPart = []any{
	map[string]any{"text": "S1"},
	map[string]any{"text": "S2"},
	map[string]any{"text": "S3"},
}

func TestEverySystemMessageReachesTheSystemInstruction(t *testing.T) {
	t.Parallel()

	server, body := recordingServer(t, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},`+
		`"finishReason":"STOP"}],"usageMetadata":{}}`)
	llm, err := New(context.Background(),
		WithAPIKey("unit-test-key"), WithEndpoint(server.URL), WithDefaultModel("gemini-2.5-flash"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(), everySystemMessage)
	require.NoError(t, err)
	require.Equal(t, everySystemPart, systemInstructionParts(t, *body))
}

func TestEverySystemMessageReachesTheCachedSystemInstruction(t *testing.T) {
	t.Parallel()

	server, body := recordingServer(t, `{"name":"cachedContents/c1","model":"models/gemini-2.5-flash"}`)
	helper, err := NewCachingHelper(context.Background(), WithAPIKey("unit-test-key"), WithEndpoint(server.URL))
	require.NoError(t, err)

	_, err = helper.CreateCachedContent(context.Background(), "gemini-2.5-flash", everySystemMessage, time.Hour, "")
	require.NoError(t, err)
	require.Equal(t, everySystemPart, systemInstructionParts(t, *body))
}
