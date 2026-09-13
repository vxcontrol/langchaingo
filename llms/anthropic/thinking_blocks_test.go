package anthropic_test

import (
	"context"
	"encoding/json"
	"fmt"
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
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type cannedMessages struct {
	mu        sync.Mutex
	responses []string
	requests  [][]byte
}

func (c *cannedMessages) serve(t *testing.T) *anthropic.LLM {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		c.mu.Lock()
		c.requests = append(c.requests, body)
		answer := c.responses[min(len(c.requests), len(c.responses))-1]
		c.mu.Unlock()
		if strings.HasPrefix(answer, "event:") {
			w.Header().Set("Content-Type", "text/event-stream")
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		_, _ = io.WriteString(w, answer)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"),
		anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-fable-5-1"))
	require.NoError(t, err)
	return llm
}

func (c *cannedMessages) assistantContent(t *testing.T, request int) []map[string]any {
	t.Helper()

	var payload struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(c.requests[request], &payload))
	for _, m := range payload.Messages {
		if m.Role == "assistant" {
			return m.Content
		}
	}
	t.Fatalf("request %d carries no assistant turn", request)
	return nil
}

func messageWith(content string) string {
	return `{"id":"msg_1","type":"message","role":"assistant","model":"claude-fable-5-1",` +
		`"content":[` + content + `],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`
}

func thinkingJSON(text, signature string) string {
	return fmt.Sprintf(`{"type":"thinking","thinking":%q,"signature":%q}`, text, signature)
}

func toolUseJSON(id string) string {
	return fmt.Sprintf(`{"type":"tool_use","id":%q,"name":"lookup","input":{"q":%q}}`, id, id)
}

func askWithTool(t *testing.T, llm *anthropic.LLM, opts ...llms.CallOption) *llms.ContentChoice {
	t.Helper()

	resp, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "look it up"),
	}, append(opts, llms.WithMaxTokens(64), llms.WithTools([]llms.Tool{lookupTool()}))...)
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	return resp.Choices[0]
}

func lookupTool() llms.Tool {
	return llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name:       "lookup",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}},
	}}
}

// replayTurn mirrors the turn pentagi builds from a choice: one text part
// carrying the reasoning, then the tool calls.
func replayTurn(t *testing.T, llm *anthropic.LLM, choice *llms.ContentChoice) {
	t.Helper()

	turn := make([]llms.ContentPart, 0, 1+len(choice.ToolCalls))
	turn = append(turn, llms.TextPartWithReasoning(choice.Content, choice.Reasoning))
	results := make([]llms.ContentPart, 0, len(choice.ToolCalls))
	for _, call := range choice.ToolCalls {
		turn = append(turn, call)
		results = append(results, llms.ToolCallResponse{ToolCallID: call.ID, Name: call.FunctionCall.Name, Content: "done"})
	}
	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "look it up"),
		{Role: llms.ChatMessageTypeAI, Parts: turn},
		{Role: llms.ChatMessageTypeTool, Parts: results},
	}, llms.WithMaxTokens(64), llms.WithTools([]llms.Tool{lookupTool()}))
	require.NoError(t, err)
}

func TestEveryThinkingBlockOfAResponseKeepsItsSignature(t *testing.T) {
	t.Parallel()

	canned := &cannedMessages{responses: []string{messageWith(
		thinkingJSON("", "sig-reasoning") + "," + thinkingJSON("checking the index", "sig-update") + "," + toolUseJSON("A"),
	)}}
	choice := askWithTool(t, canned.serve(t))

	assert.Equal(t, []reasoning.Block{
		{Signature: []byte("sig-reasoning")},
		{Text: "checking the index", Signature: []byte("sig-update")},
	}, choice.Reasoning.Sequence())
	assert.Equal(t, "checking the index", choice.Reasoning.Content)
	assert.Empty(t, choice.Reasoning.Signature, "no single signature covers two blocks")
}

func TestAThinkingBlockBetweenToolCallsKnowsItsPlace(t *testing.T) {
	t.Parallel()

	canned := &cannedMessages{responses: []string{messageWith(
		thinkingJSON("plan", "s1") + "," + toolUseJSON("A") + "," +
			thinkingJSON("next", "s2") + "," + `{"type":"redacted_thinking","data":"opaque"}` + "," + toolUseJSON("B"),
	)}}
	choice := askWithTool(t, canned.serve(t))

	assert.Equal(t, []reasoning.Block{
		{Text: "plan", Signature: []byte("s1")},
		{Text: "next", Signature: []byte("s2"), AfterToolCalls: 1},
		{Redacted: []byte("opaque"), AfterToolCalls: 1},
	}, choice.Reasoning.Sequence())
}

func TestAStreamedResponseKeepsEachBlockWithItsOwnSignature(t *testing.T) {
	t.Parallel()

	event := func(name, data string) string { return "event: " + name + "\ndata: " + data + "\n\n" }
	stream := event("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",`+
		`"model":"claude-fable-5-1","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}`) +
		event("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`) +
		event("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"first "}}`) +
		event("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"thought"}}`) +
		event("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"s1"}}`) +
		event("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		event("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":""}}`) +
		event("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"second"}}`) +
		event("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"s2"}}`) +
		event("content_block_stop", `{"type":"content_block_stop","index":1}`) +
		event("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"A","name":"lookup","input":{}}}`) +
		event("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"A\"}"}}`) +
		event("content_block_stop", `{"type":"content_block_stop","index":2}`) +
		event("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}`) +
		event("message_stop", `{"type":"message_stop"}`)

	canned := &cannedMessages{responses: []string{stream}}
	choice := askWithTool(t, canned.serve(t),
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

	assert.Equal(t, []reasoning.Block{
		{Text: "first thought", Signature: []byte("s1")},
		{Text: "second", Signature: []byte("s2")},
	}, choice.Reasoning.Sequence())
}

func TestAReplayedTurnGoesBackAsTheVendorSentIt(t *testing.T) {
	t.Parallel()

	for name, original := range map[string][]string{
		"reasoning and a progress update before one call": {
			thinkingJSON("", "sig-reasoning"), thinkingJSON("checking the index", "sig-update"), toolUseJSON("A"),
		},
		"an update before each of two calls": {
			thinkingJSON("plan", "s1"), toolUseJSON("A"), thinkingJSON("", "s2"), toolUseJSON("B"),
		},
		"an update cut off after the last call": {
			thinkingJSON("plan", "s1"), toolUseJSON("A"), thinkingJSON("", "s2"),
		},
		"an encrypted block between signed ones": {
			thinkingJSON("a", "s1"), `{"type":"redacted_thinking","data":"opaque"}`, thinkingJSON("b", "s2"), toolUseJSON("A"),
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			canned := &cannedMessages{responses: []string{messageWith(strings.Join(original, ",")), messageWith(`{"type":"text","text":"ok"}`)}}
			llm := canned.serve(t)
			replayTurn(t, llm, askWithTool(t, llm))

			var want []map[string]any
			require.NoError(t, json.Unmarshal([]byte("["+strings.Join(original, ",")+"]"), &want))
			assert.Equal(t, want, canned.assistantContent(t, 1))
		})
	}
}
