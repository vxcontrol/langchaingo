package mistral

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func generateForWarnings(t *testing.T, call ...llms.CallOption) *llms.ContentResponse {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"mistral-small-latest",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	m, err := New(WithAPIKey("k"), WithEndpoint(srv.URL), WithModel("mistral-small-latest"))
	require.NoError(t, err)

	resp, err := m.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, call...)
	require.NoError(t, err)
	return resp
}

func mistralWarningsByOption(warnings []llms.Warning) map[string]llms.Warning {
	byOption := make(map[string]llms.Warning, len(warnings))
	for _, w := range warnings {
		byOption[w.Option] = w
	}
	return byOption
}

func TestTheDoorReportsEveryIntentItCannotCarry(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithToolChoice(llms.ToolChoice{
			Type: "function", Function: &llms.FunctionReference{Name: "get_weather"},
		}),
		llms.WithStopWords([]string{"STOP"}),
		llms.WithStructuredOutput(llms.StructuredOutputConfig{
			Name: "answer", Schema: json.RawMessage(`{"type":"object"}`),
		}),
		llms.WithReasoning(llms.ReasoningHigh, 0),
	)

	got := mistralWarningsByOption(resp.Warnings)
	for option, asked := range map[string]string{
		"WithToolChoice": "get_weather", "WithStopWords": "STOP",
		"WithStructuredOutput": "answer", "WithReasoning": "high",
	} {
		w, ok := got[option]
		require.True(t, ok, "no %s warning in %v", option, resp.Warnings)
		require.Equal(t, llms.WarningDrop, w.Kind)
		require.Equal(t, asked, w.Asked)
		require.Equal(t, "mistral-small-latest", w.Model)
	}
}

func TestSwitchingThinkingOffOnMistralIsNotALoss(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithReasoningDisabled())
	require.Empty(t, resp.Warnings,
		"every mistral model disables by omission, so sending no field is the off wire, not a dropped intent")

	off := chatBodyOnTheWire(t, llms.WithSeed(7), llms.WithReasoningDisabled())
	require.NotContains(t, off, "reasoning_effort")
	require.Equal(t, chatBodyOnTheWire(t, llms.WithSeed(7)), off,
		"off by omission is the request a caller who named no reasoning sends")
}

func TestAPlainMistralCallCarriesNoWarnings(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithTemperature(0.2))
	require.Empty(t, resp.Warnings)
}

func TestTheSamplingOptionsThisDoorHasNoFieldForAreReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t,
		llms.WithTopK(40), llms.WithMinP(0.1), llms.WithMinLength(2),
		llms.WithN(3), llms.WithCandidateCount(2), llms.WithTopLogProbs(4),
		llms.WithLogProbs(true), llms.WithRepetitionPenalty(1.1),
		llms.WithFrequencyPenalty(0.5), llms.WithPresencePenalty(0.25),
	)

	got := mistralWarningsByOption(resp.Warnings)
	for option, asked := range map[string]string{
		"WithTopK": "40", "WithMinP": "0.1", "WithMinLength": "2",
		"WithN": "3", "WithCandidateCount": "2", "WithTopLogProbs": "4",
		"WithLogProbs": "true", "WithRepetitionPenalty": "1.1",
		"WithFrequencyPenalty": "0.5", "WithPresencePenalty": "0.25",
	} {
		w, ok := got[option]
		require.True(t, ok, "no %s warning in %v", option, resp.Warnings)
		require.Equal(t, llms.WarningDrop, w.Kind)
		require.Equal(t, asked, w.Asked)
	}
}

func TestExtraBodyThisDoorCannotMergeIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, llms.WithExtraBody(map[string]any{"enable_thinking": false, "chat_template_kwargs": map[string]any{}}))

	w, ok := mistralWarningsByOption(resp.Warnings)["WithExtraBody"]
	require.True(t, ok, "no extra-body warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "chat_template_kwargs, enable_thinking", w.Asked)
}

func TestTheMistralDoorReportsBothLengthOptionsItNeverReads(t *testing.T) {
	t.Parallel()

	got := mistralWarningsByOption(
		generateForWarnings(t, llms.WithMinLength(10), llms.WithMaxLength(20)).Warnings)

	for _, option := range []string{"WithMinLength", "WithMaxLength"} {
		require.Contains(t, got, option, "the door's request has no field for it")
	}
}

func TestFunctionsBesideToolsAreReportedAsDropped(t *testing.T) {
	t.Parallel()

	tools := llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name: "lookup", Parameters: map[string]any{"type": "object"},
	}}})
	functions := llms.WithFunctions([]llms.FunctionDefinition{{
		Name: "now", Parameters: map[string]any{"type": "object"},
	}})

	sentTools := func(body map[string]any) []string {
		list, _ := body["tools"].([]any)
		names := make([]string, 0, len(list))
		for _, tool := range list {
			fn, _ := tool.(map[string]any)["function"].(map[string]any)
			name, _ := fn["name"].(string)
			names = append(names, name)
		}
		return names
	}

	require.Equal(t, []string{"lookup"}, sentTools(captureMistralRequest(t, tools, functions)))
	w, ok := mistralWarningsByOption(generateForWarnings(t, tools, functions).Warnings)["WithFunctions"]
	require.True(t, ok, "the functions beside the tools went nowhere unreported")
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "1 functions", w.Asked)

	require.Equal(t, []string{"now"}, sentTools(captureMistralRequest(t, functions)))
	require.NotContains(t, mistralWarningsByOption(generateForWarnings(t, functions).Warnings), "WithFunctions")
}

func TestAToolChoiceWithoutToolsIsReportedAsOnEveryDoor(t *testing.T) {
	t.Parallel()

	toolChoiceWarnings := func(warnings []llms.Warning) []llms.Warning {
		var found []llms.Warning
		for _, w := range warnings {
			if w.Option == "WithToolChoice" {
				found = append(found, w)
			}
		}
		return found
	}
	named := llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "lookup"}}
	for _, tc := range []struct {
		choice any
		asked  string
	}{
		{"required", "any"},
		{"any", "any"},
		{named, "lookup"},
	} {
		require.NotContains(t, captureMistralRequest(t, llms.WithToolChoice(tc.choice)), "tool_choice", "%v", tc.choice)
		require.Equal(t, []llms.Warning{{
			Kind: llms.WarningDrop, Option: "WithToolChoice", Model: "mistral-small-latest",
			Asked: tc.asked, Reason: "the request carries no tools to choose from",
		}}, toolChoiceWarnings(generateForWarnings(t, llms.WithToolChoice(tc.choice)).Warnings), "%v", tc.choice)
	}

	for _, choice := range []string{"auto", "none"} {
		require.Equal(t, choice, captureMistralRequest(t, llms.WithToolChoice(choice))["tool_choice"])
		require.Empty(t, generateForWarnings(t, llms.WithToolChoice(choice)).Warnings, choice)
	}

	functions := llms.WithFunctions([]llms.FunctionDefinition{{Name: "lookup"}})
	require.Equal(t, "any", captureMistralRequest(t, functions, llms.WithToolChoice("required"))["tool_choice"])
	require.Empty(t, generateForWarnings(t, functions, llms.WithToolChoice("required")).Warnings)
	w := mistralWarningsByOption(generateForWarnings(t, functions, llms.WithToolChoice(named)).Warnings)["WithToolChoice"]
	require.Equal(t, "the door's tool choice is a bare string, so a named tool has no shape to travel in", w.Reason)
}
