package ollama

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func generateForWarnings(t *testing.T, callOpts ...llms.CallOption) *llms.ContentResponse {
	t.Helper()
	return generateForWarningsOn(t, "glm-5", callOpts...)
}

func generateForWarningsOn(t *testing.T, model string, callOpts ...llms.CallOption) *llms.ContentResponse {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"` + model + `","message":{"role":"assistant","content":"hi"},` +
			`"done":true,"done_reason":"stop"}` + "\n"))
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel(model))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, callOpts...)
	require.NoError(t, err)
	return resp
}

func ollamaWarningsByOption(warnings []llms.Warning) map[string]llms.Warning {
	byOption := make(map[string]llms.Warning, len(warnings))
	for _, w := range warnings {
		byOption[w.Option] = w
	}
	return byOption
}

func TestOptionsWithNoFieldOnThisDoorAreReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithMinP(0.05), llms.WithLogProbs(true), llms.WithTopLogProbs(2),
		llms.WithN(2), llms.WithCandidateCount(3),
		llms.WithToolChoice(llms.ToolChoice{
			Type: "function", Function: &llms.FunctionReference{Name: "get_weather"},
		}),
		llms.WithFunctions([]llms.FunctionDefinition{{Name: "now"}}))

	got := ollamaWarningsByOption(resp.Warnings)
	require.NotContains(t, got, "WithMinP", "min_p reaches the wire")
	for option, asked := range map[string]string{
		"WithLogProbs": "true", "WithTopLogProbs": "2",
		"WithN": "2", "WithCandidateCount": "3", "WithToolChoice": "get_weather",
		"WithFunctions": "1 functions",
	} {
		w, ok := got[option]
		require.True(t, ok, "no %s warning in %v", option, resp.Warnings)
		require.Equal(t, llms.WarningDrop, w.Kind)
		require.Equal(t, asked, w.Asked)
		require.Equal(t, "glm-5", w.Model)
	}
}

func TestAnEffortOllamaHasNoLevelForIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithReasoning(llms.ReasoningXHigh, 0))

	w, ok := ollamaWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningSubstitute, w.Kind)
	require.Equal(t, "xhigh", w.Asked)
	require.Equal(t, "true", w.Sent)
}

func TestAThinkingTokenBudgetHasNowhereToGoOnOllama(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithMaxTokens(4096), llms.WithReasoning(llms.ReasoningHigh, 2048))

	w, ok := ollamaWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "no reasoning warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "2048", w.Asked)
}

func TestAPlainOllamaCallCarriesNoWarnings(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithTemperature(0.2))
	require.Empty(t, resp.Warnings)
}

func TestAToolChoiceOfAutoIsNotALoss(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithToolChoice("auto"))
	require.Empty(t, resp.Warnings, "auto is what the door does with no tool choice at all")
}

func TestADroppedToolChoiceIsNamedNotNumbered(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithToolChoice("none"), llms.WithTools([]llms.Tool{{
		Type: "function", Function: &llms.FunctionDefinition{Name: "lookup", Parameters: map[string]any{"type": "object"}},
	}}))

	w, ok := ollamaWarningsByOption(resp.Warnings)["WithToolChoice"]
	require.True(t, ok, "no tool-choice warning in %v", resp.Warnings)
	require.Equal(t, "none", w.Asked)
	require.Equal(t, "the door builds no field for it", w.Reason)
}

func TestAToolChoiceWithoutToolsIsReportedAsOnEveryDoor(t *testing.T) {
	t.Parallel()

	named := llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "lookup"}}
	for _, tc := range []struct {
		choice any
		asked  string
	}{
		{"required", "any"},
		{named, "lookup"},
	} {
		resp := generateForWarnings(t, llms.WithToolChoice(tc.choice))
		require.Equal(t, []llms.Warning{{
			Kind: llms.WarningDrop, Option: "WithToolChoice", Model: "glm-5",
			Asked: tc.asked, Reason: "the request carries no tools to choose from",
		}}, resp.Warnings, "%v", tc.choice)
	}

	for _, choice := range []any{"auto", "none"} {
		resp := generateForWarnings(t, llms.WithToolChoice(choice))
		require.Empty(t, resp.Warnings, "%v: nothing to choose from changes nothing", choice)
	}
}

func TestAZeroTopKIsNotALoss(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithTopK(0))
	require.Empty(t, resp.Warnings)
}

func TestExtraBodyThisDoorCannotMergeIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithExtraBody(map[string]any{"enable_thinking": false, "chat_template_kwargs": map[string]any{}}))

	w, ok := ollamaWarningsByOption(resp.Warnings)["WithExtraBody"]
	require.True(t, ok, "no extra-body warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "chat_template_kwargs, enable_thinking", w.Asked)
}

func TestMetadataSetAfterExtraBodyStillReportsTheLoss(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithExtraBody(map[string]any{"enable_thinking": false}),
		llms.WithMetadata(map[string]any{"user": "u1"}))

	w, ok := ollamaWarningsByOption(resp.Warnings)["WithExtraBody"]
	require.True(t, ok, "no extra-body warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "enable_thinking", w.Asked)
}

func TestTheOllamaDoorReportsEachOptionOnce(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithN(2), llms.WithCandidateCount(3),
		llms.WithLogProbs(true), llms.WithTopLogProbs(5),
		llms.WithMinLength(10), llms.WithMaxLength(20),
		llms.WithVerbosity("low"), llms.WithResponseMIMEType("application/json"))

	seen := make(map[string]int, len(resp.Warnings))
	for _, w := range resp.Warnings {
		seen[w.Option]++
	}
	for option, count := range seen {
		require.Equal(t, 1, count, "%s reported %d times: %v", option, count, resp.Warnings)
	}
}

func TestTheCloudReportsTheFormatItCannotSend(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name         string
		opts         llms.CallOptions
		clientFormat string
		emulated     bool
		option       string
		kind         llms.WarningKind
		asked, sent  string
	}{
		{name: "per-call JSON mode", opts: llms.CallOptions{JSONMode: true}, option: "WithJSONMode", kind: llms.WarningDrop, asked: "true"},
		{name: "client-level format", clientFormat: "json", option: "WithFormat", kind: llms.WarningDrop, asked: "json"},
		{
			name: "an emulated schema", emulated: true, clientFormat: "json",
			opts: llms.CallOptions{JSONMode: true, StructuredOutput: &llms.StructuredOutputConfig{
				Name: "answer", Schema: json.RawMessage(ollamaSOSchema),
			}},
			option: "WithStructuredOutput", kind: llms.WarningSubstitute, asked: "answer", sent: "a prompt instruction",
		},
		{
			name: "an emulated schema without a name", emulated: true,
			opts:   llms.CallOptions{JSONMode: true, StructuredOutput: &llms.StructuredOutputConfig{Schema: json.RawMessage(ollamaSOSchema)}},
			option: "WithStructuredOutput", kind: llms.WarningSubstitute, asked: "a JSON Schema", sent: "a prompt instruction",
		},
		{name: "nothing asked"},
	} {
		warn := &llms.Warnings{}
		reportOllamaCloudFormat(warn, "gpt-oss:120b", tc.opts, tc.clientFormat, tc.emulated)
		got := warn.List()
		if tc.option == "" {
			if len(got) != 0 {
				t.Errorf("%s: want no warning, got %v", tc.name, got)
			}
			continue
		}
		if len(got) != 1 || got[0].Kind != tc.kind || got[0].Option != tc.option ||
			got[0].Asked != tc.asked || got[0].Sent != tc.sent {
			t.Errorf("%s: want one %s of %s asked %q sent %q, got %v", tc.name, tc.kind, tc.option, tc.asked, tc.sent, got)
		}
	}
}

func TestABudgetWithoutAnEffortIsReportedAsTheLevelItBecame(t *testing.T) {
	t.Parallel()

	for model, level := range map[string]string{"glm-5": "low", "gpt-oss:20b": "low"} {
		resp := generateForWarningsOn(t, model, llms.WithMaxTokens(4096), llms.WithReasoning(llms.ReasoningNone, 800))

		w, ok := ollamaWarningsByOption(resp.Warnings)["WithReasoning"]
		require.True(t, ok, model)
		require.Equal(t, llms.WarningSubstitute, w.Kind, "%s: a level went out, so nothing was dropped", model)
		require.Equal(t, "800", w.Asked, model)
		require.Equal(t, level, w.Sent, model)
	}
}

func TestABudgetWithoutAnEffortReachesTheWireAsItsLevel(t *testing.T) {
	t.Parallel()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"glm-5","message":{"role":"assistant","content":"hi"},"done":true,"done_reason":"stop"}` + "\n"))
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel("glm-5"))
	require.NoError(t, err)
	_, err = llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithMaxTokens(4096), llms.WithReasoning(llms.ReasoningNone, 800))
	require.NoError(t, err)
	require.Equal(t, "low", body["think"], "800 of 4096 tokens is the low level")
}
