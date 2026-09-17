package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func extraBodyWire(t *testing.T, model string, client []Option, answer string, opts ...llms.CallOption) map[string]json.RawMessage {
	t.Helper()

	content, err := json.Marshal(answer)
	if err != nil {
		t.Fatalf("answer: %v", err)
	}
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		if strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m",`+
				`"choices":[{"index":0,"delta":{"role":"assistant","content":`+string(content)+`},"finish_reason":"stop"}]}`+
				"\n\ndata: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":`+string(content)+`},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(append([]Option{WithBaseURL(srv.URL), WithToken("test"), WithModel(model)}, client...)...)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	if _, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...); err != nil {
		t.Fatalf("GenerateContent() error: %v", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatalf("the wire is not a JSON object: %v, body: %s", err, body)
	}
	return fields
}

func TestExtraBodyLeavesTheFieldsItDoesNotNameAsTheDoorBuiltThem(t *testing.T) {
	t.Parallel()

	seed := llms.WithSeed(9007199254740993)
	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{
		Name: "verdict",
		Schema: json.RawMessage(`{"type":"object","properties":{"reasoning":{"type":"string"},` +
			`"answer":{"type":"string"}},"required":["reasoning","answer"],"additionalProperties":false}`),
	})
	const answer = `{"reasoning":"r","answer":"a"}`
	without := extraBodyWire(t, "gpt-4o", nil, answer, seed, schema)
	with := extraBodyWire(t, "gpt-4o", nil, answer,
		seed, schema, llms.WithExtraBody(map[string]any{"preserve_thinking": true}))

	for _, field := range []string{"seed", "response_format", "messages"} {
		if string(with[field]) != string(without[field]) {
			t.Errorf("%s changed under extra body:\nwant %s\ngot  %s", field, without[field], with[field])
		}
	}
	if string(with["preserve_thinking"]) != "true" {
		t.Errorf("the extra body field did not reach the wire: %s", with["preserve_thinking"])
	}
}

func assertWireJSON(t *testing.T, fields map[string]json.RawMessage, want map[string]string) {
	t.Helper()

	for field, wantJSON := range want {
		var got, expected any
		if err := json.Unmarshal(fields[field], &got); err != nil {
			t.Errorf("%s is missing from the wire or is not JSON: %q", field, fields[field])
			continue
		}
		if err := json.Unmarshal([]byte(wantJSON), &expected); err != nil {
			t.Fatalf("want for %s is not JSON: %v", field, err)
		}
		if !reflect.DeepEqual(got, expected) {
			t.Errorf("%s on the wire:\nwant %s\ngot  %s", field, wantJSON, fields[field])
		}
	}
}

func TestExtraBodyObjectsMergeIntoTheObjectsTheDoorBuilds(t *testing.T) { //nolint:funlen // table-driven test
	t.Parallel()

	off := llms.WithReasoningDisabled()
	extra := llms.WithExtraBody
	clearThinking := extra(map[string]any{"thinking": map[string]any{"clear_thinking": false}})
	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{
		Name: "verdict",
		Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}},` +
			`"required":["a"],"additionalProperties":false}`),
	})
	stream := llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil })

	for name, tc := range map[string]struct {
		model  string
		client []Option
		answer string
		opts   []llms.CallOption
		want   map[string]string
	}{
		"glm off on its own gateway keeps clear_thinking": {
			model: "zai/glm-4.5-air", opts: []llms.CallOption{off, clearThinking},
			want: map[string]string{"thinking": `{"type":"disabled","clear_thinking":false}`},
		},
		"bare glm off keeps clear_thinking": {
			model: "glm-4.5-air", opts: []llms.CallOption{off, clearThinking},
			want: map[string]string{"thinking": `{"type":"disabled","clear_thinking":false}`},
		},
		"glm-5.2 off on its own gateway keeps clear_thinking": {
			model: "zai/glm-5.2", opts: []llms.CallOption{off, clearThinking},
			want: map[string]string{"thinking": `{"type":"disabled","clear_thinking":false}`},
		},
		"bare glm-5.2 off keeps clear_thinking": {
			model: "glm-5.2", opts: []llms.CallOption{off, clearThinking},
			want: map[string]string{"thinking": `{"type":"disabled","clear_thinking":false}`},
		},
		"kimi off on its own gateway keeps keep": {
			model: "moonshot/kimi-k2.6",
			opts:  []llms.CallOption{off, extra(map[string]any{"thinking": map[string]any{"keep": "all"}})},
			want:  map[string]string{"thinking": `{"type":"disabled","keep":"all"}`},
		},
		"bare kimi off keeps keep given as a typed map": {
			model: "kimi-k2.6",
			opts:  []llms.CallOption{off, extra(map[string]any{"thinking": map[string]string{"keep": "all"}})},
			want:  map[string]string{"thinking": `{"type":"disabled","keep":"all"}`},
		},
		"glm effort keeps its own field and gains clear_thinking": {
			model: "zai/glm-5.2", opts: []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0), clearThinking},
			want: map[string]string{"reasoning_effort": `"high"`, "thinking": `{"clear_thinking":false}`},
		},
		"claude budget keeps type and budget and gains display": {
			model: "anthropic/claude-sonnet-4-5",
			opts: []llms.CallOption{
				llms.WithReasoning(llms.ReasoningNone, 2048), llms.WithMaxTokens(8192),
				extra(map[string]any{"thinking": map[string]any{"display": "summarized"}}),
			},
			want: map[string]string{"thinking": `{"type":"enabled","budget_tokens":2048,"display":"summarized"}`},
		},
		"reasoning object keeps the effort and gains exclude": {
			model: "gpt-5", client: []Option{WithModernReasoningFormat()},
			opts: []llms.CallOption{
				llms.WithReasoning(llms.ReasoningHigh, 0),
				extra(map[string]any{"reasoning": map[string]any{"exclude": true}}),
			},
			want: map[string]string{"reasoning": `{"effort":"high","exclude":true}`},
		},
		"stream options keep include_usage": {
			model: "gpt-4o",
			opts: []llms.CallOption{
				stream, extra(map[string]any{"stream_options": map[string]any{"include_obfuscation": false}}),
			},
			want: map[string]string{"stream_options": `{"include_usage":true,"include_obfuscation":false}`},
		},
		"a nested leaf is replaced and its siblings stay": {
			model: "gpt-4o", answer: `{"a":"b"}`,
			opts: []llms.CallOption{
				schema,
				extra(map[string]any{"response_format": map[string]any{"json_schema": map[string]any{"strict": false}}}),
			},
			want: map[string]string{"response_format": `{"type":"json_schema","json_schema":{"name":"verdict",` +
				`"strict":false,"schema":{"type":"object","properties":{"a":{"type":"string"}},` +
				`"required":["a"],"additionalProperties":false}}}`},
		},
		"the caller's thinking type still wins over the door's": {
			model: "zai/glm-4.5-air",
			opts: []llms.CallOption{
				off, extra(map[string]any{"thinking": map[string]any{"type": "enabled", "clear_thinking": false}}),
			},
			want: map[string]string{"thinking": `{"type":"enabled","clear_thinking":false}`},
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			answer := tc.answer
			if answer == "" {
				answer = "ok"
			}
			assertWireJSON(t, extraBodyWire(t, tc.model, tc.client, answer, tc.opts...), tc.want)
		})
	}
}

func TestExtraBodyValuesOtherThanObjectsReplaceTheDoors(t *testing.T) {
	t.Parallel()

	off := llms.WithReasoningDisabled()
	extra := llms.WithExtraBody

	for name, tc := range map[string]struct {
		model string
		opts  []llms.CallOption
		want  map[string]string
	}{
		"a scalar": {
			model: "gpt-4o",
			opts:  []llms.CallOption{llms.WithTemperature(0.7), extra(map[string]any{"temperature": 0.9})},
			want:  map[string]string{"temperature": `0.9`},
		},
		"the DashScope flag": {
			model: "dashscope/glm-5.1",
			opts:  []llms.CallOption{off, extra(map[string]any{"enable_thinking": true})},
			want:  map[string]string{"enable_thinking": `true`},
		},
		"an array": {
			model: "gpt-4o",
			opts:  []llms.CallOption{llms.WithStopWords([]string{"a", "b"}), extra(map[string]any{"stop": []string{"c"}})},
			want:  map[string]string{"stop": `["c"]`},
		},
		"null over an object": {
			model: "zai/glm-4.5-air",
			opts:  []llms.CallOption{off, extra(map[string]any{"thinking": nil})},
			want:  map[string]string{"thinking": `null`},
		},
		"a string over an object": {
			model: "zai/glm-4.5-air",
			opts:  []llms.CallOption{off, extra(map[string]any{"thinking": "disabled"})},
			want:  map[string]string{"thinking": `"disabled"`},
		},
		"new top-level keys beside the door's": {
			model: "zai/glm-4.5-air",
			opts:  []llms.CallOption{off, extra(map[string]any{"preserve_thinking": true, "tool_choice": "auto"})},
			want: map[string]string{
				"thinking": `{"type":"disabled"}`, "preserve_thinking": `true`, "tool_choice": `"auto"`,
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			assertWireJSON(t, extraBodyWire(t, tc.model, nil, "ok", tc.opts...), tc.want)
		})
	}
}

func TestAMergedThinkingObjectReportsNoExtraBodyLoss(t *testing.T) {
	t.Parallel()

	resp := sendForWarnings(t, "zai/glm-4.5-air", llms.WithReasoningDisabled(),
		llms.WithExtraBody(map[string]any{"thinking": map[string]any{"clear_thinking": false}}))
	for _, warning := range resp.Warnings {
		if warning.Option == "WithExtraBody" {
			t.Errorf("this door merges extra body, nothing is lost, got %v", warning)
		}
	}
}

func TestExtraBodyObjectOfAnotherTypeReplacesTheDoors(t *testing.T) { //nolint:funlen // table-driven test
	t.Parallel()

	budgetAnd := func(thinking map[string]any) []llms.CallOption {
		return []llms.CallOption{
			llms.WithReasoning(llms.ReasoningNone, 2048), llms.WithMaxTokens(8192),
			llms.WithExtraBody(map[string]any{"thinking": thinking}),
		}
	}
	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{
		Name: "verdict",
		Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}},` +
			`"required":["a"],"additionalProperties":false}`),
	})
	tools := llms.WithTools([]llms.Tool{{
		Type:     "function",
		Function: &llms.FunctionDefinition{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)},
	}})
	named := llms.WithToolChoice(llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "lookup"}})
	allowedTools := map[string]any{
		"type": "allowed_tools",
		"allowed_tools": map[string]any{
			"mode":  "auto",
			"tools": []any{map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}}},
		},
	}

	for name, tc := range map[string]struct {
		model  string
		answer string
		opts   []llms.CallOption
		want   map[string]string
	}{
		"claude budget under disabled thinking": {
			model: "anthropic/claude-sonnet-4-5",
			opts:  budgetAnd(map[string]any{"type": "disabled"}),
			want:  map[string]string{"thinking": `{"type":"disabled"}`},
		},
		"claude budget under adaptive thinking": {
			model: "anthropic/claude-opus-4-6",
			opts:  budgetAnd(map[string]any{"type": "adaptive", "display": "summarized"}),
			want:  map[string]string{"thinking": `{"type":"adaptive","display":"summarized"}`},
		},
		"claude budget under the same type keeps the budget": {
			model: "anthropic/claude-sonnet-4-5",
			opts:  budgetAnd(map[string]any{"type": "enabled", "display": "omitted"}),
			want:  map[string]string{"thinking": `{"type":"enabled","budget_tokens":2048,"display":"omitted"}`},
		},
		"structured output under a json object format": {
			model: "gpt-4o", answer: `{"a":"b"}`,
			opts: []llms.CallOption{
				schema, llms.WithExtraBody(map[string]any{"response_format": map[string]any{"type": "json_object"}}),
			},
			want: map[string]string{"response_format": `{"type":"json_object"}`},
		},
		"a type the door's object does not carry merges into it": {
			model: "gpt-4o",
			opts:  []llms.CallOption{llms.WithExtraBody(map[string]any{"type": "chat"})},
			want:  map[string]string{"model": `"gpt-4o"`, "type": `"chat"`},
		},
		"a named tool choice under allowed tools": {
			model: "gpt-4o",
			opts:  []llms.CallOption{tools, named, llms.WithExtraBody(map[string]any{"tool_choice": allowedTools})},
			want: map[string]string{"tool_choice": `{"type":"allowed_tools","allowed_tools":{"mode":"auto",` +
				`"tools":[{"type":"function","function":{"name":"lookup"}}]}}`},
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			answer := tc.answer
			if answer == "" {
				answer = "ok"
			}
			assertWireJSON(t, extraBodyWire(t, tc.model, nil, answer, tc.opts...), tc.want)
		})
	}
}
