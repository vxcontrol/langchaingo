package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func sendForWarnings(t *testing.T, model string, opts ...llms.CallOption) *llms.ContentResponse {
	t.Helper()

	const completion = `{"id":"x","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completion)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithBaseURL(srv.URL), WithToken("test"), WithModel(model))
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	if err != nil {
		t.Fatalf("GenerateContent() error: %v", err)
	}
	return resp
}

func warningFor(t *testing.T, resp *llms.ContentResponse, option string) llms.Warning {
	t.Helper()

	for _, w := range resp.Warnings {
		if w.Option == option {
			return w
		}
	}
	t.Fatalf("no warning for %s, got %v", option, resp.Warnings)
	return llms.Warning{}
}

func TestTheCallerLearnsWhatTheSamplingPolicyTookAway(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "gpt-5",
		llms.WithTemperature(0.2), llms.WithTopP(0.9),
		llms.WithLogProbs(true), llms.WithTopLogProbs(3))

	temperature := warningFor(t, resp, "WithTemperature")
	if temperature.Kind != llms.WarningSubstitute || temperature.Asked != "0.2" || temperature.Sent != "1" {
		t.Errorf("temperature warning = %+v", temperature)
	}
	if temperature.Model != "gpt-5" {
		t.Errorf("temperature warning names model %q", temperature.Model)
	}

	for _, option := range []string{"WithTopP", "WithLogProbs", "WithTopLogProbs"} {
		if got := warningFor(t, resp, option); got.Kind != llms.WarningDrop || got.Sent != "" {
			t.Errorf("%s warning = %+v", option, got)
		}
	}
}

func TestAnUntouchedRequestCarriesNoWarnings(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "gpt-4o", llms.WithTemperature(0.2), llms.WithTopP(0.9))
	if len(resp.Warnings) != 0 {
		t.Errorf("want no warnings on a model that keeps sampling, got %v", resp.Warnings)
	}
}
