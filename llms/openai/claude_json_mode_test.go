package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestJSONModeStaysOffTheWireForClaude(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"anthropic/claude-sonnet-5", "claude-haiku-4-5", "us.anthropic.claude-opus-4-6-v1:0"} {
		if body := sendForWire(t, model, llms.WithJSONMode()); strings.Contains(body, "response_format") {
			t.Errorf("%s: the vendor has no json_object for this model, got body: %s", model, body)
		}
		w := warningFor(t, sendForWarnings(t, model, llms.WithJSONMode()), "WithJSONMode")
		if w.Kind != llms.WarningDrop || w.Asked != "true" || w.Sent != "" {
			t.Errorf("%s: json-mode warning = %+v", model, w)
		}
	}
}

func TestJSONModeStillReachesAModelThatTakesIt(t *testing.T) {
	t.Parallel()

	if body := sendForWire(t, "gpt-4o", llms.WithJSONMode()); !strings.Contains(body, `"response_format":{"type":"json_object"}`) {
		t.Errorf("want json_object on the wire, got body: %s", body)
	}
	if resp := sendForWarnings(t, "gpt-4o", llms.WithJSONMode()); len(resp.Warnings) != 0 {
		t.Errorf("JSON mode reached the wire, got %v", resp.Warnings)
	}
}

func TestASchemaForClaudeIsNotReportedAsALostJSONMode(t *testing.T) {
	t.Parallel()

	llm := newUnitLLM(t, WithModel("anthropic/claude-sonnet-5"))
	var opts llms.CallOptions
	llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "s", Schema: objectSchema()})(&opts)
	warn := &llms.Warnings{}
	if _, err := llm.createChatRequest(nil, opts, warn); err != nil {
		t.Fatal(err)
	}
	for _, w := range warn.List() {
		if w.Option == "WithJSONMode" {
			t.Errorf("the schema carries the JSON request, got %v", w)
		}
	}
}

func TestClientJSONFormatFollowsTheJSONModeModelRule(t *testing.T) {
	t.Parallel()

	client := []Option{WithResponseFormat(ResponseFormatJSON)}
	for _, model := range []string{
		"anthropic/claude-sonnet-5", "claude-haiku-4-5", "us.anthropic.claude-opus-4-6-v1:0", "minimax/MiniMax-M3",
	} {
		resp, sent := sendForWarningsWith(t, model, client)
		if rf, ok := sent["response_format"]; ok {
			t.Errorf("%s: the vendor takes no json_object for this model, got response_format %v", model, rf)
		}
		w := warningFor(t, resp, "WithResponseFormat")
		if w.Kind != llms.WarningDrop || w.Asked != "json_object" || w.Sent != "" || w.Model != model {
			t.Errorf("%s: response-format warning = %+v", model, w)
		}
	}

	for _, model := range []string{"MiniMax-M3", "MiniMax-M2.7"} {
		resp, sent := clientFormatOn(t, "https://api.minimax.io/v1", model)
		if rf, ok := sent["response_format"]; ok {
			t.Errorf("%s: MiniMax's API takes no response_format, got %v", model, rf)
		}
		if w := warningFor(t, resp, "WithResponseFormat"); w.Kind != llms.WarningDrop || w.Asked != "json_object" {
			t.Errorf("%s: response-format warning = %+v", model, w)
		}
	}
	if _, sent := clientFormatOn(t, "https://openrouter.ai/api/v1", "minimax/minimax-m3"); sent["response_format"] == nil {
		t.Error("OpenRouter lists response_format for minimax-m3, so the client format reaches the wire")
	}

	for _, model := range []string{"gpt-4o", "openrouter/minimax/minimax-m3"} {
		resp, sent := sendForWarningsWith(t, model, client)
		if rf, _ := sent["response_format"].(map[string]any); rf["type"] != "json_object" {
			t.Errorf("%s: want json_object on the wire, got %v", model, sent["response_format"])
		}
		if len(resp.Warnings) != 0 {
			t.Errorf("%s: the format reached the wire, got %v", model, resp.Warnings)
		}
	}

	schema := &ResponseFormat{Type: "json_schema", JSONSchema: &ResponseFormatJSONSchema{
		Name: "s", Strict: true, Schema: &ResponseFormatJSONSchemaProperty{Type: "object"},
	}}
	resp, sent := sendForWarningsWith(t, "anthropic/claude-sonnet-5", []Option{WithResponseFormat(schema)})
	if rf, _ := sent["response_format"].(map[string]any); rf["type"] != "json_schema" {
		t.Errorf("a schema is not json_object, want it on the wire, got %v", sent["response_format"])
	}
	if len(resp.Warnings) != 0 {
		t.Errorf("the schema reached the wire, got %v", resp.Warnings)
	}
}

func clientFormatOn(t *testing.T, baseURL, model string) (*llms.ContentResponse, map[string]any) {
	t.Helper()

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer), WithResponseFormat(ResponseFormatJSON))
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	require.NoError(t, err)
	var sent map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &sent))
	return resp, sent
}
