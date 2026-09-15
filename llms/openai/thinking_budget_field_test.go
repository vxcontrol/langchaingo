package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestABudgetNoFieldCarriesIsRefusedBeforeTheNetwork(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"zai/glm-5.1", "glm-4.7", "moonshot/kimi-k2.6", "minimax/MiniMax-M3"} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			var reached bool
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				reached = true
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"x","choices":[]}`)
			}))
			t.Cleanup(srv.Close)

			llm, err := New(WithBaseURL(srv.URL), WithToken("test"), WithModel(model))
			if err != nil {
				t.Fatalf("New() error: %v", err)
			}
			_, err = llm.GenerateContent(context.Background(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
				llms.WithReasoning(llms.ReasoningNone, 1024), llms.WithMaxTokens(8192))

			var want *reasoning.ErrThinkingBudgetUnsupported
			if !errors.As(err, &want) {
				t.Fatalf("a budget %s has no field for must be refused, got err=%v", model, err)
			}
			if want.Model != model {
				t.Errorf("the error names %q, want %q", want.Model, model)
			}
			if reached {
				t.Error("the refusal must come before the request leaves")
			}
		})
	}
}

func TestABudgetOnAnEffortOnlyModelTravelsAsAnEffortLikeAdaptiveClaude(t *testing.T) {
	t.Parallel()

	budget := []llms.CallOption{llms.WithReasoning(llms.ReasoningNone, 1024), llms.WithMaxTokens(8192)}
	for _, model := range []string{
		"anthropic/claude-sonnet-5", "zai/glm-5.2", "glm-5.2", "glm-5.3", "kimi-k3", "moonshot/kimi-k3",
	} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			body := captureWireWithClient(t, model, nil, budget...)
			if !strings.Contains(body, `"reasoning_effort":"low"`) {
				t.Errorf("want the level the budget maps to, got body: %s", body)
			}
			for _, field := range []string{`"thinking_budget"`, `"budget_tokens"`, `"thinking"`} {
				if strings.Contains(body, field) {
					t.Errorf("no budget field exists for this model, got %s in body: %s", field, body)
				}
			}

			w := warningFor(t, sendForWarnings(t, model, budget...), "WithReasoning")
			if w.Kind != llms.WarningDrop || w.Asked != "1024 tokens" || w.Sent != "" {
				t.Errorf("the caller must learn the budget itself did not travel, got %+v", w)
			}
		})
	}
}

func TestABudgetTheReasoningObjectCarriesIsNotRefused(t *testing.T) {
	t.Parallel()

	body := sendModernReasoningForModel(t, "zai/glm-5.1", llms.ReasoningNone, 2048)

	if !strings.Contains(body, `"reasoning":{"max_tokens":2048}`) {
		t.Errorf("the reasoning object carries the budget on this format, got body: %s", body)
	}
}
