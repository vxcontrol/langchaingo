package googleai

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func weatherCall(id, location string) string {
	if id != "" {
		id = `"id":"` + id + `",`
	}
	return `{"functionCall":{` + id + `"name":"get_weather","args":{"location":"` + location + `"}}}`
}

func streamedCandidate(finish string, parts ...string) string {
	return `data: {"candidates":[{"content":{"role":"model","parts":[` + strings.Join(parts, ",") +
		`]},"finishReason":"` + finish + `","index":0}]}` + "\r\n\r\n"
}

func TestParallelCallsOfOneFunctionGetTheirOwnIDs(t *testing.T) {
	t.Parallel()

	paris, london := weatherCall("", "Paris"), weatherCall("", "London")
	for _, tc := range []struct {
		name    string
		body    string
		wantIDs []string
	}{
		{name: "streamed in one chunk", body: streamedCandidate("STOP", paris, london)},
		{name: "streamed in two chunks", body: streamedCandidate("", paris) + streamedCandidate("STOP", london)},
		{
			name: "streamed with the vendor's IDs",
			body: streamedCandidate("", weatherCall("call-a", "Paris")) +
				streamedCandidate("STOP", weatherCall("call-b", "London")),
			wantIDs: []string{"call-a", "call-b"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			llm, err := New(t.Context(),
				WithAPIKey("unit-test-key"), WithRest(),
				WithDefaultModel("gemini-2.5-flash"),
				WithHTTPClient(&http.Client{Transport: stubStreamTransport{body: tc.body}}))
			require.NoError(t, err)

			resp, err := llm.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "weather in Paris and London?")},
				llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
			require.NoError(t, err)
			require.Len(t, resp.Choices, 1)

			requireOneIDPerCall(t, resp.Choices[0].ToolCalls, tc.wantIDs)
		})
	}

	t.Run("whole answer", func(t *testing.T) {
		t.Parallel()

		transport := &captureTransport{resp: `{"candidates":[{"content":{"role":"model","parts":[` +
			paris + `,` + london + `]},"finishReason":"STOP","index":0}]}`}
		llm, err := New(t.Context(),
			WithAPIKey("unit-test-key"), WithRest(),
			WithDefaultModel("gemini-2.5-flash"),
			WithHTTPClient(&http.Client{Transport: transport}))
		require.NoError(t, err)

		resp, err := llm.GenerateContent(t.Context(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "weather in Paris and London?")})
		require.NoError(t, err)
		require.Len(t, resp.Choices, 1)

		requireOneIDPerCall(t, resp.Choices[0].ToolCalls, nil)
	})
}

func requireOneIDPerCall(t *testing.T, calls []llms.ToolCall, wantIDs []string) {
	t.Helper()

	require.Len(t, calls, 2)
	assert.JSONEq(t, `{"location":"Paris"}`, calls[0].FunctionCall.Arguments)
	assert.JSONEq(t, `{"location":"London"}`, calls[1].FunctionCall.Arguments)
	if wantIDs != nil {
		assert.Equal(t, wantIDs, []string{calls[0].ID, calls[1].ID})
		return
	}
	for _, call := range calls {
		assert.True(t, strings.HasPrefix(call.ID, GENERATED_FUNCTION_CALL_ID_PREFIX), call.ID)
	}
	assert.NotEqual(t, calls[0].ID, calls[1].ID,
		"a tool result is matched to its call by ID, so two calls must never share one")
}
