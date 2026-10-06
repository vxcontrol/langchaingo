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

func TestBedrockSendsACacheTTLOnlyWhereTheModelTakesIt(t *testing.T) {
	t.Parallel()

	const (
		sonnet45   = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
		appProfile = "arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/a1b2c3d4e5f6"
	)
	oneHour := cacheMarkedConversation(bedrock.EphemeralCacheOneHour())
	fiveMinutes := cacheMarkedConversation(bedrock.EphemeralCache())
	converseOneHour := []any{"5m", "1h", "1h", "5m"}
	for _, tc := range []struct {
		name, model, answer string
		converse            bool
		messages            []llms.MessageContent
		wantTTLs            []any
		substituted         bool
	}{
		{"converse Sonnet 4.5", sonnet45, converseAnswer, true, oneHour, converseOneHour, false},
		{"converse unlisted Opus 6", "anthropic.claude-opus-6-v1:0", converseAnswer, true, oneHour, converseOneHour, false},
		{"converse application profile", appProfile, converseAnswer, true, oneHour, converseOneHour, false},
		{"converse unknown Claude tier", "anthropic.claude-sonata-4-v1:0", converseAnswer, true, oneHour, converseOneHour, false},
		{"converse Sonnet 4", "anthropic.claude-sonnet-4-20250514-v1:0", converseAnswer, true, oneHour, nil, true},
		{"converse Claude 3.7 Sonnet", "anthropic.claude-3-7-sonnet-20250219-v1:0", converseAnswer, true, oneHour, nil, true},
		{"converse Nova Pro", "us.amazon.nova-pro-v1:0", converseAnswer, true, oneHour, nil, true},
		{"converse Sonnet 4 five minutes", "anthropic.claude-sonnet-4-20250514-v1:0", converseAnswer, true, fiveMinutes, nil, false},
		{"legacy Sonnet 4.5", sonnet45, legacyAnswer, false, oneHour, []any{"1h", "1h"}, false},
		{"legacy Opus 4.1", "us.anthropic.claude-opus-4-1-20250805-v1:0", legacyAnswer, false, oneHour, nil, true},
		{"legacy Opus 4.1 five minutes", "us.anthropic.claude-opus-4-1-20250805-v1:0", legacyAnswer, false, fiveMinutes, nil, false},
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
			resp, err := bedrockLLMAgainst(t, srv, opts...).GenerateContent(t.Context(), tc.messages)
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

			var substitutes []llms.Warning
			for _, w := range resp.Warnings {
				if w.Option == "WithCacheControl" {
					substitutes = append(substitutes, w)
				}
			}
			if !tc.substituted {
				require.Empty(t, substitutes)
				return
			}
			require.Len(t, substitutes, 1, "one warning however many marks asked for an hour")
			require.Equal(t, llms.WarningSubstitute, substitutes[0].Kind)
			require.Equal(t, "1h", substitutes[0].Asked)
			require.Equal(t, "5m", substitutes[0].Sent)
		})
	}
}

func cacheMarkedConversation(mark *llms.CacheControl) []llms.MessageContent {
	return []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "system prompt"),
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{
			bedrock.WithCacheControl(llms.TextPart("first question"), mark),
		}},
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			bedrock.WithCacheControl(llms.TextPart("first answer"), mark),
		}},
		llms.TextParts(llms.ChatMessageTypeHuman, "second question"),
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
