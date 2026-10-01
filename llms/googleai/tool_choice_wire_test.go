package googleai

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestTheCallersToolChoiceReachesTheWire(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name    string
		choice  any
		mode    string
		allowed string
	}{
		{name: "required becomes ANY", choice: "required", mode: `"mode":"ANY"`},
		{name: "auto becomes AUTO", choice: "auto", mode: `"mode":"AUTO"`},
		{name: "none becomes NONE", choice: "none", mode: `"mode":"NONE"`},
		{
			name:    "a named tool becomes ANY over that name",
			choice:  llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "get_weather"}},
			mode:    `"mode":"ANY"`,
			allowed: `"allowedFunctionNames":["get_weather"]`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var body string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				body = string(b)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},`+
					`"finishReason":"STOP"}],"usageMetadata":{}}`)
			}))
			defer server.Close()

			llm, err := New(context.Background(),
				WithAPIKey("unit-test-key"), WithEndpoint(server.URL),
				WithDefaultModel("gemini-2.5-flash"))
			require.NoError(t, err)

			_, err = llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what is the weather?")},
				llms.WithTools(weatherTool()), llms.WithToolChoice(tc.choice))
			require.NoError(t, err)

			assert.Contains(t, body, `"toolConfig"`, "the door must send a tool config at all")
			assert.Contains(t, body, tc.mode)
			if tc.allowed != "" {
				assert.Contains(t, body, tc.allowed)
			}
		})
	}
}

func TestNoToolChoiceLeavesTheRequestAlone(t *testing.T) {
	t.Parallel()

	var body string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{}}`)
	}))
	defer server.Close()

	llm, err := New(context.Background(),
		WithAPIKey("unit-test-key"), WithEndpoint(server.URL),
		WithDefaultModel("gemini-2.5-flash"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what is the weather?")},
		llms.WithTools(weatherTool()))
	require.NoError(t, err)

	assert.NotContains(t, body, `"toolConfig"`,
		"a caller who asked for nothing must not have a mode chosen for them")
}

func weatherTool() []llms.Tool {
	return []llms.Tool{{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "get_weather",
			Description: "report the weather in a city",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
			},
		},
	}}
}

func TestAReplayedCallWithoutArgumentsReachesTheWire(t *testing.T) {
	t.Parallel()

	for _, arguments := range []string{"null", "", "  "} {
		t.Run(fmt.Sprintf("%q", arguments), func(t *testing.T) {
			t.Parallel()

			var body string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				body = string(b)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},`+
					`"finishReason":"STOP"}],"usageMetadata":{}}`)
			}))
			t.Cleanup(server.Close)

			llm, err := New(t.Context(), WithAPIKey("unit-test-key"), WithEndpoint(server.URL),
				WithDefaultModel("gemini-2.5-flash"))
			require.NoError(t, err)

			history := []llms.MessageContent{
				llms.TextParts(llms.ChatMessageTypeHuman, "what time is it?"),
				{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.ToolCall{
					ID: "c1", Type: "function",
					FunctionCall: &llms.FunctionCall{Name: "clock", Arguments: arguments},
				}}},
				{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{llms.ToolCallResponse{
					ToolCallID: "c1", Name: "clock", Content: "noon",
				}}},
			}
			_, err = llm.GenerateContent(t.Context(), history)
			require.NoError(t, err, "a call the model made without arguments must replay")
			assert.Contains(t, body, `"name":"clock"`)
		})
	}
}
