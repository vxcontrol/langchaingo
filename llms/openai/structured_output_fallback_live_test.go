package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
	"github.com/vxcontrol/langchaingo/llms/structuredoutput"
)

// The vendors' OpenAI-compatible endpoints are fixed rather than read from
// ZAI_API_BASE or MINIMAX_API_BASE because a replay matches the recorded URL
// byte for byte.
const (
	zaiBaseURL     = "https://api.z.ai/api/paas/v4"
	miniMaxBaseURL = "https://api.minimax.io/v1"
)

func newTestZAIClient(t *testing.T, opts ...Option) *LLM {
	t.Helper()
	return newTestVendorClient(t, "ZAI_API_KEY", zaiBaseURL, opts...)
}

func newTestMiniMaxClient(t *testing.T, opts ...Option) *LLM {
	t.Helper()
	return newTestVendorClient(t, "MINIMAX_API_KEY", miniMaxBaseURL, opts...)
}

func newTestVendorClient(t *testing.T, keyEnv, baseURL string, opts ...Option) *LLM {
	t.Helper()

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, keyEnv)
	rr := httprr.OpenForTest(t, http.DefaultTransport)

	clientOpts := []Option{WithBaseURL(baseURL), WithHTTPClient(rr.Client())}
	if rr.Recording() {
		clientOpts = append(clientOpts, WithToken(os.Getenv(keyEnv)))
	} else {
		clientOpts = append(clientOpts, WithToken("fake-api-key-for-testing"))
	}
	llm, err := New(append(clientOpts, opts...)...)
	require.NoError(t, err)
	return llm
}

// cityGuideSchema is a nested schema inside OpenAI's strict subset, so the same
// schema would go to OpenAI as json_schema.
const cityGuideSchema = `{"type":"object","properties":{` +
	`"city":{"type":"string"},` +
	`"landmarks":{"type":"array","items":{"type":"object","properties":{` +
	`"name":{"type":"string"},"kind":{"type":"string","enum":["museum","monument","park","church","other"]}},` +
	`"required":["name","kind"],"additionalProperties":false}}},` +
	`"required":["city","landmarks"],"additionalProperties":false}`

type cityGuide struct {
	City      string `json:"city"`
	Landmarks []struct {
		Name string `json:"name"`
		Kind string `json:"kind"`
	} `json:"landmarks"`
}

// askForACityGuide runs the fallback against a live vendor with a system prompt
// that demands Markdown, which the schema instruction has to override.
func askForACityGuide(t *testing.T, llm *LLM, stream bool, opts ...llms.CallOption) {
	t.Helper()

	opts = append(opts, llms.WithStructuredOutput(llms.StructuredOutputConfig{
		Name: "city_guide", Schema: json.RawMessage(cityGuideSchema),
	}))
	var streamed strings.Builder
	if stream {
		opts = append(opts, llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			if chunk.Type == streaming.ChunkTypeText {
				streamed.WriteString(chunk.Content)
			}
			return nil
		}))
	}

	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "You are a travel guide. Answer in Markdown with headings."),
		llms.TextParts(llms.ChatMessageTypeHuman, "Name two landmarks of Paris."),
	}, opts...)
	require.NoError(t, err, "the answer must validate against the schema")

	var guide cityGuide
	require.NoError(t, json.Unmarshal([]byte(resp.Choices[0].Content), &guide))
	assert.Equal(t, "Paris", guide.City)
	assert.NotEmpty(t, guide.Landmarks)
	if stream {
		// MiniMax streams its thinking inside <think> tags in the text as well.
		_, text := reasoning.SplitContent(streamed.String())
		assert.Equal(t, resp.Choices[0].Content, structuredoutput.UnwrapFencedJSON(text))
	}

	var substitutes int
	for _, w := range resp.Warnings {
		if w.Option == "WithStructuredOutput" && w.Kind == llms.WarningSubstitute {
			substitutes++
		}
	}
	assert.Equal(t, 1, substitutes)
}

func TestStructuredOutputFallbackDeepSeek(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "non-streaming"
		if stream {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) {
			llm := newTestDeepSeekClient(t, WithModel("deepseek-flash"), WithStructuredOutputFallback())
			askForACityGuide(t, llm, stream)
		})
	}
}

func TestStructuredOutputFallbackZAI(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "non-streaming"
		if stream {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) {
			llm := newTestZAIClient(t, WithModel("glm-5.3-flash"), WithStructuredOutputFallback())
			askForACityGuide(t, llm, stream, llms.WithReasoning(llms.ReasoningLow, 0))
		})
	}
}

func TestStructuredOutputFallbackMiniMax(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "non-streaming"
		if stream {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) {
			llm := newTestMiniMaxClient(t, WithModel("MiniMax-M2.7"), WithStructuredOutputFallback())
			askForACityGuide(t, llm, stream)
		})
	}
}

// TestStructuredOutputFallbackZAIToolRound checks that the schema instruction
// leaves tool calls open: the model calls the tool first, then answers with JSON
// that matches the schema once the tool result is in.
func TestStructuredOutputFallbackZAIToolRound(t *testing.T) {
	llm := newTestZAIClient(t, WithModel("glm-5.3-flash"), WithStructuredOutputFallback(), WithPreserveReasoningContent())
	askForTheWeatherWithATool(t, llm, llms.WithReasoning(llms.ReasoningLow, 0))
}

// TestStructuredOutputFallbackMiniMaxToolRound does the same on MiniMax, whose
// thinking goes back inside <think> tags at the head of the tool-call turn.
func TestStructuredOutputFallbackMiniMaxToolRound(t *testing.T) {
	llm := newTestMiniMaxClient(t, WithModel("MiniMax-M2.7"), WithStructuredOutputFallback(), WithPreserveReasoningContent())
	askForTheWeatherWithATool(t, llm)
}

func askForTheWeatherWithATool(t *testing.T, llm *LLM, extra ...llms.CallOption) {
	t.Helper()

	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{
		Name: "weather",
		Schema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},` +
			`"temperature_c":{"type":"integer"},"summary":{"type":"string"}},` +
			`"required":["city","temperature_c","summary"],"additionalProperties":false}`),
	})
	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "get_weather", Description: "Current weather for a city",
		Parameters: map[string]any{
			"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}},
			"required": []string{"city"},
		},
	}}})
	opts := append([]llms.CallOption{schema, tools}, extra...)

	messages := make([]llms.MessageContent, 0, 3)
	messages = append(messages, llms.TextParts(llms.ChatMessageTypeHuman, "What is the weather in Paris? Use the tool."))
	first, err := llm.GenerateContent(context.Background(), messages, opts...)
	require.NoError(t, err)
	require.NotEmpty(t, first.Choices[0].ToolCalls, "the schema instruction must not keep the model from calling the tool")
	call := first.Choices[0].ToolCalls[0]

	assistant := llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
		llms.TextContent{Text: first.Choices[0].Content, Reasoning: first.Choices[0].Reasoning}, call,
	}}
	messages = append(messages, assistant, llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
		llms.ToolCallResponse{ToolCallID: call.ID, Name: call.FunctionCall.Name, Content: `{"temperature_c":18,"conditions":"light rain"}`},
	}})

	final, err := llm.GenerateContent(context.Background(), messages, opts...)
	require.NoError(t, err, "the answer after the tool result must validate against the schema")
	var weather struct {
		City         string `json:"city"`
		TemperatureC int    `json:"temperature_c"`
	}
	require.NoError(t, json.Unmarshal([]byte(final.Choices[0].Content), &weather))
	assert.Equal(t, "Paris", weather.City)
	assert.Equal(t, 18, weather.TemperatureC)
}
