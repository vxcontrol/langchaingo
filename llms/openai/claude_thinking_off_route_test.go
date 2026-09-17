package openai

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestTurningOffClaudeIsRefusedWhereTheHostDocumentsNoThinkingObject(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct {
		baseURL, model string
		clientOpts     []Option
	}{
		"the gateway's deepinfra route":  {gatewayBaseURL, "deepinfra/anthropic/claude-opus-5", nil},
		"the gateway's perplexity route": {gatewayBaseURL, "perplexity/anthropic/claude-sonnet-5", nil},
		"the gateway's openrouter route": {gatewayBaseURL, "openrouter/anthropic/claude-sonnet-5", nil},
		"DeepInfra":                      {"http://api.deepinfra.com/v1/openai", "anthropic/claude-opus-5", nil},
		"Perplexity":                     {"http://api.perplexity.ai", "anthropic/claude-sonnet-5", nil},
		"OpenRouter":                     {"http://openrouter.ai/api/v1", "anthropic/claude-sonnet-5", nil},
		"the reasoning object format": {
			gatewayBaseURL, "anthropic/claude-sonnet-5", []Option{WithModernReasoningFormat()},
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			sent, err := sendOffToHost(t, tc.baseURL, tc.model, tc.clientOpts)

			var unsupported *reasoning.ErrReasoningOffUnsupported
			if !errors.As(err, &unsupported) {
				t.Errorf("want ErrReasoningOffUnsupported, got err=%v", err)
			}
			if sent {
				t.Error("the refusal must come before the network, but a request went out")
			}
		})
	}
}

func sendOffToHost(t *testing.T, baseURL, model string, clientOpts []Option) (bool, error) {
	t.Helper()

	sent := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		sent = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	addr := srv.Listener.Addr().String()
	transport := &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, addr)
	}}
	t.Cleanup(transport.CloseIdleConnections)

	opts := append([]Option{
		WithBaseURL(baseURL), WithToken("token"), WithModel(model),
		WithHTTPClient(&http.Client{Transport: transport}),
	}, clientOpts...)
	llm, err := New(opts...)
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithReasoningDisabled())
	return sent, err
}
