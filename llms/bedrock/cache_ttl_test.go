package bedrock_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestBedrockSendsACacheTTLOnlyToModelsThatTakeTheOneHourTTL(t *testing.T) {
	t.Parallel()

	messages := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "system prompt"),
		llms.TextParts(llms.ChatMessageTypeHuman, "first question"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			bedrock.WithCacheControl(llms.TextPart("first answer"), bedrock.EphemeralCacheOneHour()),
		}},
		llms.TextParts(llms.ChatMessageTypeHuman, "second question"),
	}
	for _, tc := range []struct {
		name, model, answer string
		converse            bool
		wantTTLs            []any
	}{
		{"converse Sonnet 4.5", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", converseAnswer, true, []any{"5m", "1h", "5m"}},
		{"converse unlisted Opus 6", "anthropic.claude-opus-6-v1:0", converseAnswer, true, []any{"5m", "1h", "5m"}},
		{"converse Sonnet 4", "anthropic.claude-sonnet-4-20250514-v1:0", converseAnswer, true, nil},
		{"converse Claude 3.7 Sonnet", "anthropic.claude-3-7-sonnet-20250219-v1:0", converseAnswer, true, nil},
		{"converse Nova Pro", "us.amazon.nova-pro-v1:0", converseAnswer, true, nil},
		{"legacy Sonnet 4.5", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", legacyAnswer, false, []any{"1h"}},
		{"legacy Opus 4.1", "us.anthropic.claude-opus-4-1-20250805-v1:0", legacyAnswer, false, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var raw []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.answer)
			}))
			t.Cleanup(srv.Close)
			opts := []bedrock.Option{bedrock.WithModel(tc.model)}
			if tc.converse {
				opts = append(opts, bedrock.WithConverseAPI())
			}
			resp, err := bedrockLLMAgainst(t, srv, opts...).GenerateContent(t.Context(), messages)
			require.NoError(t, err)

			key := "cache_control"
			if tc.converse {
				key = "cachePoint"
			}
			var ttls []any
			points := 0
			for _, mark := range collectJSONObjects(t, raw, key) {
				points++
				if ttl, ok := mark["ttl"]; ok {
					ttls = append(ttls, ttl)
				}
			}
			require.NotZero(t, points, "%s", raw)
			require.ElementsMatch(t, tc.wantTTLs, ttls, "%s", raw)

			w, warned := bedrockWarningsByOption(resp.Warnings)["WithCacheControl"]
			if tc.wantTTLs != nil {
				require.False(t, warned, "%v", resp.Warnings)
				return
			}
			require.True(t, warned, "%v", resp.Warnings)
			require.Equal(t, llms.WarningSubstitute, w.Kind)
			require.Equal(t, "1h", w.Asked)
			require.Equal(t, "5m", w.Sent)
		})
	}
}

func collectJSONObjects(t *testing.T, raw []byte, key string) []map[string]any {
	t.Helper()

	var body any
	require.NoError(t, json.Unmarshal(raw, &body))
	var found []map[string]any
	var walk func(any)
	walk = func(v any) {
		switch v := v.(type) {
		case map[string]any:
			if object, ok := v[key].(map[string]any); ok {
				found = append(found, object)
			}
			for _, child := range v {
				walk(child)
			}
		case []any:
			for _, child := range v {
				walk(child)
			}
		}
	}
	walk(body)
	return found
}
