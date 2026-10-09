package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type responsesDoer struct {
	answer string
	body   []byte
	path   string
}

func (d *responsesDoer) Do(req *http.Request) (*http.Response, error) {
	d.body, _ = io.ReadAll(req.Body)
	d.path = req.URL.Path
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(d.answer)),
	}, nil
}

func routedCall(t *testing.T, baseURL, model string, opts ...llms.CallOption) (string, map[string]any) {
	t.Helper()

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
	_, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "look it up")}, opts...)
	require.NoError(t, err, "%s on %s", model, baseURL)
	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	return strings.TrimPrefix(doer.path, "/v1"), body
}

func TestOnlyWhatChatCompletionsCannotCarryGoesToOpenAIsResponses(t *testing.T) {
	t.Parallel()

	const openAI = "https://api.openai.com/v1"
	tools := llms.WithTools([]llms.Tool{astraTool()})
	high := llms.WithReasoning(llms.ReasoningHigh, 0)
	for name, tc := range map[string]struct {
		baseURL, model string
		opts           []llms.CallOption
		path           string
	}{
		"gpt-6-luna with tools":             {openAI, "gpt-6-luna", []llms.CallOption{tools}, "/responses"},
		"gpt-5.6-terra with tools":          {openAI, "gpt-5.6-terra", []llms.CallOption{tools}, "/responses"},
		"gpt-5.6-terra asked for reasoning": {openAI, "gpt-5.6-terra", []llms.CallOption{tools, high}, "/responses"},
		"gpt-5.6-terra without tools":       {openAI, "gpt-5.6-terra", nil, "/chat/completions"},
		"gpt-5.6-terra thinking off":        {openAI, "gpt-5.6-terra", []llms.CallOption{tools, llms.WithReasoningDisabled()}, "/chat/completions"},
		"gpt-5.6-terra at effort none": {openAI, "gpt-5.6-terra", []llms.CallOption{
			tools, llms.WithReasoning(llms.ReasoningEffort(reasoning.OpenAIDisableEffort), 0),
		}, "/chat/completions"},
		"gpt-5.4-mini with tools":          {openAI, "gpt-5.4-mini", []llms.CallOption{tools}, "/chat/completions"},
		"gpt-5.4-mini asked for reasoning": {openAI, "gpt-5.4-mini", []llms.CallOption{tools, high}, "/responses"},
		"gpt-5.5 asked for reasoning":      {openAI, "gpt-5.5", []llms.CallOption{tools, high}, "/responses"},
		"gpt-6-astra with tools":           {openAI, "gpt-6-astra", []llms.CallOption{tools}, "/responses"},
		"gpt-6-astra without tools":        {openAI, "gpt-6-astra", nil, "/chat/completions"},
		"gpt-4.1 with tools":               {openAI, "gpt-4.1", []llms.CallOption{tools}, "/chat/completions"},
		"a gateway":                        {"http://litellm.internal/v1", "gpt-5.6-terra", []llms.CallOption{tools}, "/chat/completions"},
		"OpenRouter":                       {"https://openrouter.ai/api/v1", "openai/gpt-5.6-terra", []llms.CallOption{tools}, "/chat/completions"},
		"Azure":                            {"https://pentagi.openai.azure.com/openai/v1", "gpt-5.6-terra", []llms.CallOption{tools}, "/chat/completions"},
	} {
		path, body := routedCall(t, tc.baseURL, tc.model, tc.opts...)
		require.True(t, strings.HasSuffix(path, tc.path), "%s: %s", name, path)
		require.Equal(t, tc.model, body["model"], name)
		if tc.path == "/responses" {
			require.Equal(t, false, body["store"], name)
		}
	}

	path, body := routedCall(t, openAI, "gpt-4.1", tools, llms.WithModel("gpt-6-astra"))
	require.Equal(t, "/responses", path)
	require.Equal(t, "gpt-6-astra", body["model"])
}

func TestAToolTurnGoesBackAsResponsesItems(t *testing.T) {
	t.Parallel()

	const model = "gpt-6-luna"
	thought := reasoning.FromBlocks([]reasoning.Block{
		{ID: "rs_1", Text: "Scan first.", Redacted: []byte("enc-1")},
		{ID: "rs_2", Redacted: []byte("enc-2"), AfterToolCalls: 1},
	}).WrittenBy(model)
	claude := reasoning.FromBlocks([]reasoning.Block{{Text: "earlier", Signature: []byte("sig")}}).WrittenBy("claude-opus-5-5")
	history := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "rules"),
		llms.TextParts(llms.ChatMessageTypeHuman, "Hello."),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("Hi.", claude)}},
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{
			llms.TextPart("Scan host A."), llms.ImageURLContent{URL: "https://example.com/net.png", Detail: "low"},
		}},
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.TextPartWithReasoning("Starting a scan.", thought),
			llms.ToolCall{ID: "call_1", Type: "function", FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: `{"host":"A"}`}},
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "call_1", Name: "lookup", Content: "22/tcp open"},
		}},
	}

	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithModel(model), WithHTTPClient(doer))
	_, err := llm.GenerateContent(context.Background(), history, llms.WithTools([]llms.Tool{astraTool()}),
		llms.WithToolChoice(llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "lookup"}}),
		llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithMaxTokens(2048))
	require.NoError(t, err)
	require.Equal(t, "/v1/responses", doer.path)

	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	require.Equal(t, []any{
		map[string]any{"type": "message", "role": "system", "content": "rules"},
		map[string]any{"type": "message", "role": "user", "content": "Hello."},
		map[string]any{"type": "message", "role": "assistant", "content": "Hi.", "phase": "final_answer"},
		map[string]any{"type": "message", "role": "user", "content": []any{
			map[string]any{"type": "input_text", "text": "Scan host A."},
			map[string]any{"type": "input_image", "image_url": "https://example.com/net.png", "detail": "low"},
		}},
		map[string]any{"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-1",
			"summary": []any{map[string]any{"type": "summary_text", "text": "Scan first."}}},
		map[string]any{"type": "message", "role": "assistant", "content": "Starting a scan.", "phase": "commentary"},
		map[string]any{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": `{"host":"A"}`},
		map[string]any{"type": "reasoning", "id": "rs_2", "encrypted_content": "enc-2", "summary": []any{}},
		map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "22/tcp open"},
	}, body["input"])
	require.Equal(t, map[string]any{"effort": "high"}, body["reasoning"])
	require.Equal(t, map[string]any{"type": "function", "name": "lookup"}, body["tool_choice"])
	require.InDelta(t, 2048, body["max_output_tokens"], 0)
	require.NotContains(t, body, "messages")
	require.NotContains(t, body, "max_completion_tokens")
}

const responsesToolAnswer = `{"id":"resp_1","object":"response","model":"gpt-6-luna","status":"completed","output":[` +
	`{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Scan first."}],"encrypted_content":"enc-1"},` +
	`{"type":"message","id":"msg_1","status":"completed","role":"assistant","phase":"commentary",` +
	`"content":[{"type":"output_text","text":"Starting a scan.","annotations":[]}]},` +
	`{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"host\":\"A\"}","status":"completed"},` +
	`{"type":"reasoning","id":"rs_2","summary":[],"encrypted_content":"enc-2"}],` +
	`"usage":{"input_tokens":81,"input_tokens_details":{"cached_tokens":64,"cache_write_tokens":8},` +
	`"output_tokens":1035,"output_tokens_details":{"reasoning_tokens":832},"total_tokens":1116}}`

func TestAResponsesAnswerComesBackAsTheChoiceAndGoesBackAsItCame(t *testing.T) {
	t.Parallel()

	const model = "gpt-6-luna"
	doer := &responsesDoer{answer: responsesToolAnswer}
	llm := newUnitLLM(t, WithModel(model), WithHTTPClient(doer))
	question := llms.TextParts(llms.ChatMessageTypeHuman, "Scan host A.")
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{question},
		llms.WithTools([]llms.Tool{astraTool()}))
	require.NoError(t, err)

	choice := resp.Choices[0]
	require.Equal(t, "Starting a scan.", choice.Content)
	require.Equal(t, "tool_calls", choice.StopReason)
	require.Equal(t, []llms.ToolCall{{ID: "call_1", Type: "function",
		FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: `{"host":"A"}`}}}, choice.ToolCalls)
	require.Equal(t, model, choice.Reasoning.Model)
	require.Equal(t, []reasoning.Block{
		{ID: "rs_1", Text: "Scan first.", Redacted: []byte("enc-1")},
		{ID: "rs_2", Redacted: []byte("enc-2"), AfterToolCalls: 1},
	}, choice.Reasoning.Sequence())
	for key, want := range map[string]int{
		"PromptTokens": 81, "CompletionTokens": 1035, "TotalTokens": 1116, "ReasoningTokens": 832,
		"CacheReadInputTokens": 64, "CacheCreationInputTokens": 8,
	} {
		require.Equal(t, want, choice.GenerationInfo[key], key)
	}

	next := []llms.MessageContent{question, choice.Message(), {Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
		llms.ToolCallResponse{ToolCallID: "call_1", Name: "lookup", Content: "22/tcp open"},
	}}}
	_, err = llm.GenerateContent(context.Background(), next, llms.WithTools([]llms.Tool{astraTool()}))
	require.NoError(t, err)
	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	input := body["input"].([]any)
	types := make([]string, 0, len(input))
	for _, item := range input {
		types = append(types, item.(map[string]any)["type"].(string))
	}
	require.Equal(t, []string{"message", "reasoning", "message", "function_call", "reasoning", "function_call_output"}, types,
		"the output items in the order they came, then the tool result")
}

func wireReasoning(id string) map[string]any {
	return map[string]any{"type": "reasoning", "id": id, "summary": []any{}, "encrypted_content": "enc-" + id}
}

func wireMessage(phase, text string) map[string]any {
	return map[string]any{"type": "message", "role": "assistant", "content": text, "phase": phase}
}

func wireCall(id string) map[string]any {
	return map[string]any{"type": "function_call", "call_id": id, "name": "lookup", "arguments": `{}`}
}

func TestEachAssistantItemGoesBackInItsPlaceWithItsPhase(t *testing.T) {
	t.Parallel()

	for name, items := range map[string][]map[string]any{
		"a preamble and the answer": {
			wireReasoning("rs_1"), wireMessage("commentary", "I'll inspect the logs."),
			wireReasoning("rs_2"), wireMessage("final_answer", "Root cause: race."),
		},
		"reasoning after the preamble": {
			wireReasoning("rs_1"), wireMessage("commentary", "Checking."), wireReasoning("rs_2"), wireCall("call_1"),
		},
		"a preamble alone": {wireMessage("commentary", "Working on it.")},
		"a preamble before each call": {
			wireReasoning("rs_1"), wireMessage("commentary", "Scanning A."), wireCall("call_1"),
			wireReasoning("rs_2"), wireMessage("commentary", "Scanning B."), wireCall("call_2"),
		},
	} {
		require.Equal(t, items, replayedAssistantItems(t, items), name)
	}
}

func replayedAssistantItems(t *testing.T, items []map[string]any) []map[string]any {
	t.Helper()

	output := make([]any, 0, len(items))
	for _, item := range items {
		if item["type"] == "message" {
			item = map[string]any{"type": "message", "id": "msg", "status": "completed", "role": "assistant", "phase": item["phase"],
				"content": []any{map[string]any{"type": "output_text", "text": item["content"], "annotations": []any{}}}}
		}
		output = append(output, item)
	}
	answer, err := json.Marshal(map[string]any{"id": "resp_1", "model": "gpt-6-luna", "status": "completed", "output": output})
	require.NoError(t, err)

	doer := &responsesDoer{answer: string(answer)}
	llm := newUnitLLM(t, WithModel("gpt-6-luna"), WithHTTPClient(doer))
	question := llms.TextParts(llms.ChatMessageTypeHuman, "Why did it fail?")
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{question}, llms.WithTools([]llms.Tool{astraTool()}))
	require.NoError(t, err)

	stored, err := json.Marshal(resp.Choices[0].Message())
	require.NoError(t, err)
	var replayed llms.MessageContent
	require.NoError(t, json.Unmarshal(stored, &replayed))
	calls := resp.Choices[0].ToolCalls
	next := make([]llms.MessageContent, 0, 2+len(calls))
	next = append(next, question, replayed)
	for _, call := range calls {
		next = append(next, llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: call.ID, Name: "lookup", Content: "done"},
		}})
	}
	_, err = llm.GenerateContent(context.Background(), next, llms.WithTools([]llms.Tool{astraTool()}))
	require.NoError(t, err)

	var body struct {
		Input []map[string]any `json:"input"`
	}
	require.NoError(t, json.Unmarshal(doer.body, &body))
	require.Equal(t, map[string]any{"type": "message", "role": "user", "content": "Why did it fail?"}, body.Input[0])
	results := body.Input[len(body.Input)-len(calls):]
	for i, call := range calls {
		require.Equal(t, map[string]any{"type": "function_call_output", "call_id": call.ID, "output": "done"}, results[i])
	}
	return body.Input[1 : len(body.Input)-len(calls)]
}

func TestATruncatedResponsesAnswerIsReportedAsTruncated(t *testing.T) {
	t.Parallel()

	doer := &responsesDoer{answer: `{"id":"resp_1","model":"gpt-6-luna","status":"incomplete",` +
		`"incomplete_details":{"reason":"max_output_tokens"},"output":[]}`}
	llm := newUnitLLM(t, WithModel("gpt-6-luna"), WithHTTPClient(doer))
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithTools([]llms.Tool{astraTool()}))
	require.NoError(t, err)
	require.True(t, resp.Choices[0].Truncated)
}

func TestTheResponsesRequestReportsWhatItHasNoFieldFor(t *testing.T) {
	t.Parallel()

	tools := llms.WithTools([]llms.Tool{astraTool()})
	doer := &bodyDoer{}
	llm := newUnitLLM(t, WithModel("gpt-6-luna"), WithHTTPClient(doer))
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		tools, llms.WithSeed(7), llms.WithN(2))
	require.NoError(t, err)
	dropped := map[string]string{}
	for _, w := range resp.Warnings {
		if w.Kind == llms.WarningDrop {
			dropped[w.Option] = w.Asked
			require.Equal(t, "gpt-6-luna", w.Model)
		}
	}
	require.Equal(t, "7", dropped["WithSeed"])
	require.Equal(t, "2", dropped["WithN"])
	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	require.NotContains(t, body, "seed")
	require.NotContains(t, body, "n")

	doer = &bodyDoer{}
	llm = newUnitLLM(t, WithModel("gpt-6-luna"), WithHTTPClient(doer))
	resp, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		tools, llms.WithN(1))
	require.NoError(t, err)
	require.Empty(t, resp.Warnings, "one choice is all the Responses API returns")
	var single map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &single))
	require.NotContains(t, single, "n")

	doer = &bodyDoer{}
	llm = newUnitLLM(t, WithModel("gpt-6-luna"), WithHTTPClient(doer))
	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		tools, llms.WithStopWords([]string{"END"}))
	var stop *reasoning.ErrStopWordsUnsupported
	require.ErrorAs(t, err, &stop)
	require.Nil(t, doer.body, "refused before the network")
}

func TestALiteLLMPassThroughToOpenAIIsOpenAIsOwnAPI(t *testing.T) {
	t.Parallel()

	tools := llms.WithTools([]llms.Tool{astraTool()})
	for baseURL, path := range map[string]string{ //nolint:gosec
		"https://llm.pentagi.net/openai/v1":             "/openai/v1/responses",
		"https://llm.pentagi.net/openai/v1/":            "/openai/v1/responses",
		"https://llm.pentagi.net/openai_passthrough/v1": "/openai_passthrough/v1/responses",
		"https://llm.pentagi.net/v1":                    "/v1/chat/completions",
		"https://llm.pentagi.net/openai/deployments/x":  "/openai/deployments/x/chat/completions",
		"https://openrouter.ai/api/v1":                  "/api/v1/chat/completions",
		"https://pentagi.openai.azure.com/openai/v1":    "/openai/v1/chat/completions",
	} {
		doer := &bodyDoer{}
		llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel("gpt-5.6-terra"), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, tools)
		require.NoError(t, err, baseURL)
		require.Equal(t, path, doer.path, baseURL)
	}
}

func TestResponsesCarriesStructuredOutputAndVerbosityInText(t *testing.T) {
	t.Parallel()

	schema := `{"type":"object","properties":{"host":{"type":"string"}},"required":["host"],"additionalProperties":false}`
	doer := &bodyDoer{content: `{"host":"A"}`}
	llm := newUnitLLM(t, WithModel("gpt-6-luna"), WithHTTPClient(doer))
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "scan")},
		llms.WithTools([]llms.Tool{astraTool()}), llms.WithVerbosity("low"),
		llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "scan", Schema: json.RawMessage(schema)}))
	require.NoError(t, err)
	require.Equal(t, `{"host":"A"}`, resp.Choices[0].Content)
	require.Equal(t, "/v1/responses", doer.path)

	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	var wantSchema any
	require.NoError(t, json.Unmarshal([]byte(schema), &wantSchema))
	require.Equal(t, map[string]any{
		"format":    map[string]any{"type": "json_schema", "name": "scan", "schema": wantSchema, "strict": true},
		"verbosity": "low",
	}, body["text"])
	require.NotContains(t, body, "response_format")
}

func TestAStreamedResponsesAnswerReachesTheCallerAsItComes(t *testing.T) {
	t.Parallel()

	var final bytes.Buffer
	require.NoError(t, json.Compact(&final, []byte(responsesToolAnswer)))
	doer := &responsesDoer{answer: "event: response.output_text.delta\n" +
		`data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"Starting "}` + "\n\n" +
		"event: response.output_text.delta\n" +
		`data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"a scan."}` + "\n\n" +
		"event: response.output_item.done\n" +
		`data: {"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"host\":\"A\"}"}}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"type":"response.completed","response":` + final.String() + "}\n\n"}
	llm := newUnitLLM(t, WithModel("gpt-6-luna"), WithHTTPClient(doer))

	var text strings.Builder
	var calls []streaming.ToolCall
	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "scan")},
		llms.WithTools([]llms.Tool{astraTool()}),
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			text.WriteString(chunk.Content)
			if chunk.Type == streaming.ChunkTypeToolCall {
				calls = append(calls, chunk.ToolCall)
			}
			return nil
		}))
	require.NoError(t, err)

	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	require.Equal(t, true, body["stream"])
	require.Equal(t, "Starting a scan.", text.String())
	require.Equal(t, []streaming.ToolCall{streaming.NewToolCall("call_1", "lookup", `{"host":"A"}`)}, calls)
	require.Equal(t, "Starting a scan.", resp.Choices[0].Content)
	require.Len(t, resp.Choices[0].Reasoning.Sequence(), 2)
	require.Equal(t, 1116, resp.Choices[0].GenerationInfo["TotalTokens"])
}
