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
)

func cacheMarks(t *testing.T, clientOpts []anthropic.Option, callOpts ...llms.CallOption) []string {
	t.Helper()

	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-5",`+
			`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(append([]anthropic.Option{
		anthropic.WithToken("test-key"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-sonnet-5"),
	}, clientOpts...)...)
	require.NoError(t, err)
	_, err = llm.GenerateContent(t.Context(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "rules"),
		llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
	}, append([]llms.CallOption{llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}})}, callOpts...)...)
	require.NoError(t, err)

	var sent struct {
		Tools    []map[string]any `json:"tools"`
		System   json.RawMessage  `json:"system"`
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(body, &sent))
	var marks []string
	for _, tool := range sent.Tools {
		if tool["cache_control"] != nil {
			marks = append(marks, "tools")
		}
	}
	var system []map[string]any
	if json.Unmarshal(sent.System, &system) == nil {
		for _, block := range system {
			if block["cache_control"] != nil {
				marks = append(marks, "system")
			}
		}
	}
	for _, msg := range sent.Messages {
		for _, block := range msg.Content {
			if block["cache_control"] != nil {
				marks = append(marks, "message")
			}
		}
	}
	return marks
}

func TestACallStrategyReplacesTheClients(t *testing.T) {
	t.Parallel()

	everything := anthropic.WithDefaultCacheStrategy(anthropic.CacheStrategy{
		CacheTools: true, CacheSystem: true, CacheMessages: true, TTL: "5m",
	})

	require.Equal(t, []string{"tools", "system", "message"}, cacheMarks(t, []anthropic.Option{everything}),
		"without a call strategy the client's applies as before")
	require.Equal(t, []string{"system"}, cacheMarks(t, []anthropic.Option{everything},
		anthropic.WithCacheStrategy(anthropic.CacheStrategy{CacheSystem: true})))
	require.Empty(t, cacheMarks(t, []anthropic.Option{everything}, anthropic.WithCacheStrategy(anthropic.CacheStrategy{})),
		"a one-off call turns the client's markers off")
}
