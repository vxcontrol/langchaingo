package anthropicclient

import (
	"context"
	"net/http"
	"os"
	"testing"

	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms/streaming"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClient_CreateCompletion(t *testing.T) {
	ctx := t.Context()
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "ANTHROPIC_API_KEY")

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		apiKey = "test-api-key"
	}

	client, err := New(apiKey, "claude-2.1", DefaultBaseURL, WithHTTPClient(rr.Client()))
	require.NoError(t, err)

	client.UseLegacyTextCompletionsAPI = true

	req := &CompletionRequest{
		Model:       "claude-2.1",
		Prompt:      "\n\nHuman: Hello, how are you?\n\nAssistant:",
		Temperature: 0.0,
		MaxTokens:   100,
	}

	resp, err := client.CreateCompletion(ctx, req)
	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.NotEmpty(t, resp.Text)
}

func TestClient_CreateMessage(t *testing.T) {
	ctx := t.Context()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	apiKey := "test-api-key"
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" && rr.Recording() {
		apiKey = key
	}

	client, err := New(apiKey, "claude-3-opus-20240229", DefaultBaseURL, WithHTTPClient(rr.Client()))
	require.NoError(t, err)

	req := &MessageRequest{
		Model: "claude-3-opus-20240229",
		Messages: []ChatMessage{
			{
				Role: "user",
				Content: []Content{
					TextContent{
						Type: "text",
						Text: "Hello, how are you?",
					},
				},
			},
		},
		MaxTokens: 100,
	}

	resp, err := client.CreateMessage(ctx, req)
	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.NotEmpty(t, resp.Content)
}

func TestClient_CreateMessageStream(t *testing.T) {
	ctx := t.Context()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	apiKey := "test-api-key"
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" && rr.Recording() {
		apiKey = key
	}

	client, err := New(apiKey, "claude-3-opus-20240229", DefaultBaseURL, WithHTTPClient(rr.Client()))
	require.NoError(t, err)

	var (
		chunks     []string
		streamDone bool
	)
	req := &MessageRequest{
		Model: "claude-3-opus-20240229",
		Messages: []ChatMessage{
			{
				Role: "user",
				Content: []Content{
					TextContent{
						Type: "text",
						Text: "Count from 1 to 5",
					},
				},
			},
		},
		MaxTokens: 100,
		Stream:    true,
		StreamingFunc: func(_ context.Context, chunk streaming.Chunk) error {
			switch chunk.Type {
			case streaming.ChunkTypeText:
				chunks = append(chunks, chunk.Content)
			case streaming.ChunkTypeDone:
				streamDone = true
			default:
				// ignore other chunk types
			}
			return nil
		},
	}

	resp, err := client.CreateMessage(ctx, req)
	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.True(t, streamDone)
	assert.NotEmpty(t, chunks)
}

func TestClient_WithAnthropicBetaHeader(t *testing.T) {
	ctx := t.Context()

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()

	apiKey := "test-api-key"
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" && rr.Recording() {
		apiKey = key
	}

	client, err := New(apiKey, "claude-3-opus-20240229", DefaultBaseURL,
		WithHTTPClient(rr.Client()),
		WithAnthropicBetaHeader("tools-2024-05-16"),
	)
	require.NoError(t, err)

	req := &MessageRequest{
		Model: "claude-3-opus-20240229",
		Messages: []ChatMessage{
			{
				Role: "user",
				Content: []Content{
					TextContent{
						Type: "text",
						Text: "What's the weather like?",
					},
				},
			},
		},
		MaxTokens: 100,
		Tools: []Tool{
			{
				Name:        "get_weather",
				Description: "Get the weather for a location",
				InputSchema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "The location to get weather for",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	resp, err := client.CreateMessage(ctx, req)
	require.NoError(t, err)
	assert.NotNil(t, resp)
}
