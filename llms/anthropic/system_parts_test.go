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

func sentSystem(t *testing.T, messages []llms.MessageContent, opts ...llms.CallOption) (any, int, error) {
	t.Helper()

	var (
		body     []byte
		requests int
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-5",` +
			`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(
		anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL),
		anthropic.WithModel("claude-sonnet-5"),
	)
	require.NoError(t, err)

	if _, err := llm.GenerateContent(t.Context(), messages, opts...); err != nil {
		return nil, requests, err
	}
	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	return payload["system"], requests, nil
}

func TestEverySystemPartReachesTheWire(t *testing.T) {
	t.Parallel()

	system := func(parts ...llms.ContentPart) llms.MessageContent {
		return llms.MessageContent{Role: llms.ChatMessageTypeSystem, Parts: parts}
	}
	block := func(text string) map[string]any { return map[string]any{"type": "text", "text": text} }
	cachedBlock := func(text, ttl string) map[string]any {
		return map[string]any{"type": "text", "text": text, "cache_control": map[string]any{"type": "ephemeral", "ttl": ttl}}
	}
	human := llms.TextParts(llms.ChatMessageTypeHuman, "hi")

	for _, tc := range []struct {
		name     string
		messages []llms.MessageContent
		want     any
	}{
		{
			name:     "a lone text stays the plain string",
			messages: []llms.MessageContent{system(llms.TextPart("S1")), human},
			want:     "S1",
		},
		{
			name:     "two system messages are two blocks, not one glued string",
			messages: []llms.MessageContent{system(llms.TextPart("S1")), system(llms.TextPart("S2")), human},
			want:     []any{block("S1"), block("S2")},
		},
		{
			name:     "every part of one system message is sent",
			messages: []llms.MessageContent{system(llms.TextPart("PART-A "), llms.TextPart("PART-B")), human},
			want:     []any{block("PART-A "), block("PART-B")},
		},
		{
			name: "a plain system message next to a cached one is kept",
			messages: []llms.MessageContent{
				system(llms.TextPart("PLAIN-S1")),
				system(anthropic.WithCacheControl(llms.TextPart("CACHED-S2"), anthropic.EphemeralCacheOneHour())),
				human,
			},
			want: []any{block("PLAIN-S1"), cachedBlock("CACHED-S2", "1h")},
		},
		{
			name:     "an empty text adds nothing",
			messages: []llms.MessageContent{system(llms.TextPart("S1")), system(llms.TextPart("")), human},
			want:     "S1",
		},
		{
			name:     "no system message sends no system",
			messages: []llms.MessageContent{human},
			want:     nil,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, _, err := sentSystem(t, tc.messages)
			require.NoError(t, err)
			require.Equal(t, tc.want, got)
		})
	}
}

func TestMalformedSystemMessageFailsBeforeTheRequest(t *testing.T) {
	t.Parallel()

	human := llms.TextParts(llms.ChatMessageTypeHuman, "hi")
	image := llms.ImageURLContent{URL: "https://example.com/a.png"}

	for _, tc := range []struct {
		name   string
		system llms.MessageContent
		want   error
	}{
		{"no parts", llms.MessageContent{Role: llms.ChatMessageTypeSystem}, anthropic.ErrEmptySystemMessage},
		{"an image", llms.MessageContent{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{image}}, anthropic.ErrInvalidContentType},
		{
			"a cached image",
			llms.MessageContent{Role: llms.ChatMessageTypeSystem, Parts: []llms.ContentPart{
				llms.TextPart("S1"), anthropic.WithCacheControl(image, anthropic.EphemeralCache()),
			}},
			anthropic.ErrInvalidContentType,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, requests, err := sentSystem(t, []llms.MessageContent{tc.system, human})
			require.ErrorIs(t, err, tc.want)
			require.Zero(t, requests)
		})
	}
}

func TestCachedSystemStrategyMarksTheLastSystemBlock(t *testing.T) {
	t.Parallel()

	messages := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "S1"),
		llms.TextParts(llms.ChatMessageTypeSystem, "S2"),
		llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
	}
	got, _, err := sentSystem(t, messages, anthropic.WithCacheStrategy(anthropic.CacheStrategy{CacheSystem: true}))
	require.NoError(t, err)
	require.Equal(t, []any{
		map[string]any{"type": "text", "text": "S1"},
		map[string]any{"type": "text", "text": "S2", "cache_control": map[string]any{"type": "ephemeral", "ttl": "5m"}},
	}, got)
}
