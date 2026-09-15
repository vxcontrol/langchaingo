package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
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

func TestTheThinkingWireOfGLMKimiAndMiniMaxOnEachHost(t *testing.T) {
	t.Parallel()

	budget := func(tokens int) []llms.CallOption {
		return []llms.CallOption{llms.WithReasoning(llms.ReasoningNone, tokens), llms.WithMaxTokens(8192)}
	}
	off := []llms.CallOption{llms.WithReasoningDisabled()}

	type row struct {
		model   string
		call    []llms.CallOption
		refused bool
		want    map[string]any
	}
	refusedModels := []string{
		"glm-5.1", "zai/glm-5.1", "kimi-k2.6", "moonshot/kimi-k2.6", "MiniMax-M3", "minimax/MiniMax-M3",
	}
	budgets := []int{1024, 2500, 6000}
	rows := make([]row, 0, len(refusedModels)*len(budgets)+6)
	for _, model := range refusedModels {
		for _, tokens := range budgets {
			rows = append(rows, row{model: model, call: budget(tokens), refused: true})
		}
	}
	rows = append(rows,
		row{model: "zai/glm-5.2", call: budget(1024), want: map[string]any{"reasoning_effort": "low"}},
		row{model: "glm-5.2", call: budget(6000), want: map[string]any{"reasoning_effort": "high"}},
		row{model: "dashscope/glm-5.1", call: budget(1024), want: map[string]any{"thinking_budget": float64(1024)}},
		row{model: "dashscope/kimi-k2.7-code", call: budget(1024), want: map[string]any{"thinking_budget": float64(1024)}},
		row{model: "glm-5.1", call: off, want: map[string]any{"thinking": map[string]any{"type": "disabled"}}},
		row{model: "zai/glm-5.1", call: off, want: map[string]any{"thinking": map[string]any{"type": "disabled"}}},
	)

	for _, tc := range rows {
		var raw []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			raw, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m",`+
				`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
		}))
		llm, err := New(WithBaseURL(srv.URL), WithToken("test"), WithModel(tc.model))
		if err != nil {
			t.Fatalf("New() error: %v", err)
		}
		_, err = llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, tc.call...)
		srv.Close()

		var refusal *reasoning.ErrThinkingBudgetUnsupported
		if tc.refused {
			if !errors.As(err, &refusal) || raw != nil {
				t.Errorf("%s: want ErrThinkingBudgetUnsupported before the network, got err=%v body=%s",
					tc.model, err, raw)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: GenerateContent() error: %v", tc.model, err)
			continue
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("%s: decode body: %v", tc.model, err)
		}
		got := map[string]any{}
		for _, key := range []string{"thinking", "reasoning_effort", "reasoning", "enable_thinking", "thinking_budget"} {
			if value, ok := body[key]; ok {
				got[key] = value
			}
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: reasoning fields on the wire = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestABudgetTheReasoningObjectCarriesIsNotRefused(t *testing.T) {
	t.Parallel()

	body := sendModernReasoningForModel(t, "zai/glm-5.1", llms.ReasoningNone, 2048)

	if !strings.Contains(body, `"reasoning":{"max_tokens":2048}`) {
		t.Errorf("the reasoning object carries the budget on this format, got body: %s", body)
	}
}
