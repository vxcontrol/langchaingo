package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type modelAPI struct {
	mu           sync.Mutex
	capabilities string
	lookups      []string
	messages     []map[string]any
}

func (m *modelAPI) serve(t *testing.T, opts ...anthropic.Option) *anthropic.LLM {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.mu.Lock()
		defer m.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if id, ok := strings.CutPrefix(r.URL.Path, "/models/"); ok && r.Method == http.MethodGet {
			m.lookups = append(m.lookups, id)
			if m.capabilities == "" {
				w.WriteHeader(http.StatusNotFound)
				_, _ = io.WriteString(w, `{"type":"error","error":{"type":"not_found_error","message":"not found"}}`)
				return
			}
			_, _ = io.WriteString(w, `{"type":"model","id":"claude-test","display_name":"Claude","created_at":"2026-10-01T00:00:00Z",`+
				`"max_input_tokens":200000,"max_tokens":64000,"capabilities":`+m.capabilities+`}`)
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		m.messages = append(m.messages, body)
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m",`+
			`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(append([]anthropic.Option{anthropic.WithToken("test-key"), anthropic.WithBaseURL(srv.URL)}, opts...)...)
	require.NoError(t, err)
	return llm
}

func (m *modelAPI) lastMessage(t *testing.T) map[string]any {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	require.NotEmpty(t, m.messages, "no message request reached the API")
	return m.messages[len(m.messages)-1]
}

func capabilities(thinking, adaptive, enabled string, disabled *bool, effortLevels []string, structured bool) string {
	levels := map[string]bool{}
	for _, l := range effortLevels {
		levels[l] = true
	}
	flag := func(on bool) map[string]bool { return map[string]bool{"supported": on} }
	types := map[string]any{"adaptive": flag(adaptive == "yes"), "enabled": flag(enabled == "yes")}
	if disabled != nil {
		types["disabled"] = flag(*disabled)
	}
	caps := map[string]any{
		"batch": flag(true), "citations": flag(true), "code_execution": flag(true),
		"context_management": flag(false), "image_input": flag(true), "pdf_input": flag(true),
		"structured_outputs": flag(structured),
		"thinking":           map[string]any{"supported": thinking == "yes", "types": types},
		"effort": map[string]any{
			"supported": len(effortLevels) > 0, "low": flag(levels["low"]), "medium": flag(levels["medium"]),
			"high": flag(levels["high"]), "xhigh": flag(levels["xhigh"]), "max": flag(levels["max"]),
		},
	}
	data, _ := json.Marshal(caps)
	return string(data)
}

func boolPtr(v bool) *bool { return &v }

func ask(t *testing.T, llm *anthropic.LLM, model string, opts ...llms.CallOption) (*llms.ContentResponse, error) {
	t.Helper()
	return llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		append([]llms.CallOption{llms.WithModel(model), llms.WithMaxTokens(16000)}, opts...)...)
}

func warningFor(resp *llms.ContentResponse, option string, kind llms.WarningKind) (llms.Warning, bool) {
	for _, w := range resp.Warnings {
		if w.Option == option && w.Kind == kind {
			return w, true
		}
	}
	return llms.Warning{}, false
}

func TestAModelOutsideTheTablesTakesTheEffortLevelsTheAPIReports(t *testing.T) {
	t.Parallel()

	api := &modelAPI{capabilities: capabilities("yes", "yes", "no", boolPtr(true), []string{"low", "medium", "high"}, true)}
	llm := api.serve(t)

	resp, err := ask(t, llm, "claude-sonnet-6", llms.WithReasoning(llms.ReasoningMax, 0))
	require.NoError(t, err)

	body := api.lastMessage(t)
	assert.Equal(t, map[string]any{"type": "adaptive", "display": "summarized"}, body["thinking"])
	assert.Equal(t, map[string]any{"effort": "high"}, body["output_config"],
		"the inherited Sonnet 5.5 row would send max; the API lists high as the top level")
	inherit, ok := warningFor(resp, "WithModel", llms.WarningInherit)
	require.True(t, ok)
	assert.Contains(t, inherit.Reason, "what the API reports about the model")
	assert.Equal(t, []string{"claude-sonnet-6"}, api.lookups)
}

func TestAModelTheAPIReportsAsBudgetOnlyGetsBudgetThinking(t *testing.T) {
	t.Parallel()

	api := &modelAPI{capabilities: capabilities("yes", "no", "yes", boolPtr(true), []string{"low", "medium", "high"}, true)}
	llm := api.serve(t)

	_, err := ask(t, llm, "claude-opus-6", llms.WithAdaptiveReasoning(llms.ReasoningHigh), llms.WithReasoning(llms.ReasoningHigh, 4096))
	require.NoError(t, err)

	body := api.lastMessage(t)
	assert.Equal(t, map[string]any{"type": "enabled", "budget_tokens": float64(4096)}, body["thinking"],
		"the inherited Opus 5.5 row is adaptive only; the API reports only budget thinking")
	assert.Equal(t, map[string]any{"effort": "high"}, body["output_config"], "the API reports effort with budget thinking")
}

func TestAModelTheAPIReportsAsNotThinkingIsSentNoThinking(t *testing.T) {
	t.Parallel()

	api := &modelAPI{capabilities: capabilities("no", "no", "no", boolPtr(true), nil, true)}
	llm := api.serve(t)

	resp, err := ask(t, llm, "claude-haiku-6", llms.WithReasoning(llms.ReasoningHigh, 0))
	require.NoError(t, err)

	body := api.lastMessage(t)
	assert.NotContains(t, body, "thinking")
	assert.NotContains(t, body, "output_config")
	dropped, ok := warningFor(resp, "WithReasoning", llms.WarningDrop)
	require.True(t, ok, "%v", resp.Warnings)
	assert.Equal(t, "high", dropped.Asked)
}

func TestAnEffortTheAPIReportsNoLevelsForIsDroppedWithAWarning(t *testing.T) {
	t.Parallel()

	api := &modelAPI{capabilities: capabilities("yes", "yes", "no", boolPtr(true), nil, true)}
	llm := api.serve(t)

	resp, err := ask(t, llm, "claude-opus-6", llms.WithReasoning(llms.ReasoningHigh, 0))
	require.NoError(t, err)

	body := api.lastMessage(t)
	assert.Equal(t, map[string]any{"type": "adaptive", "display": "summarized"}, body["thinking"])
	assert.NotContains(t, body, "output_config")
	dropped, ok := warningFor(resp, "WithReasoning", llms.WarningDrop)
	require.True(t, ok, "%v", resp.Warnings)
	assert.Equal(t, "high", dropped.Asked)
}

func TestTheOffSwitchFollowsWhatTheAPIReportsAboutDisabled(t *testing.T) {
	t.Parallel()

	api := &modelAPI{capabilities: capabilities("yes", "yes", "no", boolPtr(true), []string{"low", "medium", "high"}, true)}
	_, err := ask(t, api.serve(t), "claude-opus-6", llms.WithReasoningDisabled())
	require.NoError(t, err)
	assert.Equal(t, map[string]any{"type": "disabled"}, api.lastMessage(t)["thinking"],
		"the inherited Opus 5.5 row refuses to turn thinking off; the API reports that disabled is accepted")

	api = &modelAPI{capabilities: capabilities("yes", "yes", "no", boolPtr(false), []string{"low", "medium", "high"}, true)}
	_, err = ask(t, api.serve(t), "claude-sonnet-6", llms.WithReasoningDisabled())
	require.NoError(t, err)
	assert.Equal(t, map[string]any{"type": "between_tools"}, api.lastMessage(t)["thinking"],
		"a model that rejects disabled keeps the Sonnet line's lowest setting")

	api = &modelAPI{capabilities: capabilities("yes", "yes", "no", nil, []string{"low", "medium", "high"}, true)}
	resp, err := ask(t, api.serve(t), "claude-opus-6", llms.WithReasoningDisabled())
	require.NoError(t, err, "an answer that does not say leaves the off switch to the inherited row")
	assert.Equal(t, map[string]any{"type": "disabled"}, api.lastMessage(t)["thinking"])
	_, softened := warningFor(resp, "WithReasoningDisabled", llms.WarningInherit)
	assert.True(t, softened, "the inherited refusal goes out with a warning: %v", resp.Warnings)

	api = &modelAPI{capabilities: capabilities("yes", "yes", "no", boolPtr(false), []string{"low", "medium", "high"}, true)}
	_, err = ask(t, api.serve(t), "claude-nova-1", llms.WithReasoningDisabled())
	var refusal *reasoning.ErrReasoningOffUnsupported
	require.ErrorAs(t, err, &refusal)
	assert.Equal(t, []string{"claude-nova-1"}, api.lookups)
	assert.Empty(t, api.messages, "the refusal comes before the message request")
}

func TestStructuredOutputTheAPIReportsAsUnsupportedIsRefusedBeforeTheRequest(t *testing.T) {
	t.Parallel()

	api := &modelAPI{capabilities: capabilities("yes", "yes", "no", boolPtr(true), []string{"low", "medium", "high"}, false)}
	_, err := ask(t, api.serve(t), "claude-opus-6", llms.WithStructuredOutput(llms.StructuredOutputConfig{
		Name:   "answer",
		Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`),
	}))
	var unsupported *llms.ErrStructuredOutputUnsupported
	require.ErrorAs(t, err, &unsupported)
	assert.Empty(t, api.messages)
}

func TestTheAPIIsNotAskedAboutAModelTheTablesList(t *testing.T) {
	t.Parallel()

	api := &modelAPI{capabilities: capabilities("no", "no", "no", nil, nil, false)}
	llm := api.serve(t)

	_, err := ask(t, llm, "claude-opus-5-5", llms.WithReasoning(llms.ReasoningHigh, 0))
	require.NoError(t, err)
	assert.Empty(t, api.lookups)
	assert.Equal(t, map[string]any{"effort": "high"}, api.lastMessage(t)["output_config"], "the documented row decides")

	api = &modelAPI{capabilities: capabilities("no", "no", "no", nil, nil, false)}
	_, err = ask(t, api.serve(t, anthropic.WithoutModelLookup()), "claude-opus-6", llms.WithReasoning(llms.ReasoningHigh, 0))
	require.NoError(t, err)
	assert.Empty(t, api.lookups, "WithoutModelLookup keeps the door on the tables")
	assert.Equal(t, map[string]any{"effort": "high"}, api.lastMessage(t)["output_config"])
}

func TestAHostWithoutTheModelEndpointLeavesTheModelToTheTables(t *testing.T) {
	t.Parallel()

	api := &modelAPI{}
	llm := api.serve(t)

	for range 2 {
		resp, err := ask(t, llm, "claude-sonnet-6", llms.WithReasoning(llms.ReasoningMax, 0))
		require.NoError(t, err)
		inherit, ok := warningFor(resp, "WithModel", llms.WarningInherit)
		require.True(t, ok)
		assert.Contains(t, inherit.Reason, "claude-sonnet-5-5")
		assert.NotContains(t, inherit.Reason, "what the API reports")
	}
	assert.Equal(t, map[string]any{"effort": "max"}, api.lastMessage(t)["output_config"], "the inherited row decides")
	assert.Equal(t, []string{"claude-sonnet-6"}, api.lookups, "a failed lookup is not repeated within the TTL")
}

func TestACallWithoutACapabilityDecisionDoesNotAskTheAPI(t *testing.T) {
	t.Parallel()

	api := &modelAPI{capabilities: capabilities("yes", "yes", "no", boolPtr(true), []string{"low", "medium", "high"}, true)}
	_, err := ask(t, api.serve(t), "claude-sonnet-6")
	require.NoError(t, err)
	assert.Empty(t, api.lookups)
}
