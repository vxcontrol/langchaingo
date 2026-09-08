package mistral

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	sdk "github.com/gage-technologies/mistral-go"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func chatBodyOnTheWire(t *testing.T, call ...llms.CallOption) map[string]any {
	t.Helper()

	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"mistral-small-latest",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	require.NoError(t, err)
	_, err = m.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, call...)
	require.NoError(t, err)

	var body map[string]any
	require.NoError(t, json.Unmarshal(raw, &body))
	return body
}

func TestACallWithoutToolsDoesNotAnnounceTools(t *testing.T) {
	t.Parallel()

	body := chatBodyOnTheWire(t)

	assert.NotContains(t, body, "tools",
		"a caller that named no tool must not have one announced for it")
}

func TestACallWithToolsCarriesThem(t *testing.T) {
	t.Parallel()

	tools := []llms.Tool{{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "rooms_free",
			Description: "how many rooms are free",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"floor": map[string]any{"type": "integer"}},
			},
		},
	}}

	body := chatBodyOnTheWire(t, llms.WithTools(tools))

	carried, ok := body["tools"].([]any)
	require.True(t, ok, "the tools the caller named reach the wire")
	require.Len(t, carried, 1)
}

func TestUsageSurvivesAChunkThatDoesNotCarryIt(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, chunk := range []string{
			`{"id":"x","object":"chat.completion.chunk","created":17,"model":"mistral-small-latest",` +
				`"choices":[{"index":0,"delta":{"role":"assistant","content":"an answer"},"finish_reason":""}],` +
				`"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			`{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}`,
		} {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
			if flusher != nil {
				flusher.Flush()
			}
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	require.NoError(t, err)

	sink := func(context.Context, streaming.Chunk) error { return nil }
	resp, err := m.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithStreamingFunc(sink))
	require.NoError(t, err)

	info := resp.Choices[0].GenerationInfo
	usage, ok := info["usage"].(sdk.UsageInfo)
	require.True(t, ok, "the usage the vendor sent reaches the caller")
	assert.Equal(t, 8, usage.TotalTokens,
		"a later chunk without usage must not erase the count the vendor already gave")
	assert.Equal(t, 17, info["created"],
		"the same holds for the fields that ride along with it")
}
