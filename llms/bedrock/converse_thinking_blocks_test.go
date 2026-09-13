package bedrock_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const converseClaude = "anthropic.claude-sonnet-4-5-20250929-v1:0"

type converseRecorder struct {
	mu        sync.Mutex
	responses []string
	requests  [][]byte
}

func (c *converseRecorder) serve(t *testing.T) *bedrock.LLM {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		c.mu.Lock()
		c.requests = append(c.requests, body)
		answer := c.responses[min(len(c.requests), len(c.responses))-1]
		c.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, answer)
	}))
	t.Cleanup(srv.Close)

	return bedrockLLMAgainst(t, srv, bedrock.WithModel(converseClaude), bedrock.WithConverseAPI())
}

func converseReply(blocks ...string) string {
	return `{"output":{"message":{"role":"assistant","content":[` + strings.Join(blocks, ",") + `]}},` +
		`"stopReason":"tool_use","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`
}

func converseThought(text, signature string) string {
	return fmt.Sprintf(`{"reasoningContent":{"reasoningText":{"text":%q,"signature":%q}}}`, text, signature)
}

func converseEncrypted(data string) string {
	return fmt.Sprintf(`{"reasoningContent":{"redactedContent":%q}}`, base64.StdEncoding.EncodeToString([]byte(data)))
}

func converseToolUse(id string) string {
	return fmt.Sprintf(`{"toolUse":{"toolUseId":%q,"name":"lookup","input":{"q":%q}}}`, id, id)
}

func converseShape(t *testing.T, content []map[string]any) []string {
	t.Helper()

	shape := make([]string, 0, len(content))
	for _, block := range content {
		switch {
		case block["reasoningContent"] != nil:
			thought, _ := block["reasoningContent"].(map[string]any)
			if text, ok := thought["reasoningText"].(map[string]any); ok {
				shape = append(shape, fmt.Sprintf("thought %v/%v", text["text"], text["signature"]))
			} else {
				shape = append(shape, fmt.Sprintf("encrypted %v", thought["redactedContent"]))
			}
		case block["toolUse"] != nil:
			use, _ := block["toolUse"].(map[string]any)
			shape = append(shape, fmt.Sprintf("tool %v", use["toolUseId"]))
		case block["text"] != nil:
			shape = append(shape, fmt.Sprintf("text %v", block["text"]))
		default:
			t.Fatalf("unexpected content block %v", block)
		}
	}
	return shape
}

func (c *converseRecorder) replayedAssistant(t *testing.T, request int) []string {
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
			return converseShape(t, m.Content)
		}
	}
	t.Fatalf("request %d carries no assistant turn", request)
	return nil
}

func lookupTools() []llms.Tool {
	return []llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
		Name:       "lookup",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}},
	}}}
}

func askConverse(t *testing.T, llm *bedrock.LLM, opts ...llms.CallOption) *llms.ContentChoice {
	t.Helper()

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "look it up")},
		append(opts, llms.WithTools(lookupTools()))...)
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	return resp.Choices[0]
}

func TestConverseKeepsEveryReasoningBlockWithItsSignature(t *testing.T) {
	t.Parallel()

	rec := &converseRecorder{responses: []string{converseReply(
		converseThought("", "sig-reasoning"), converseThought("checking the index", "sig-update"), converseToolUse("A"),
	)}}
	choice := askConverse(t, rec.serve(t))

	assert.Equal(t, []reasoning.Block{
		{Text: "", Signature: []byte("sig-reasoning")},
		{Text: "checking the index", Signature: []byte("sig-update")},
	}, choice.Reasoning.Sequence())
	assert.Empty(t, choice.Reasoning.Signature, "no single signature covers two blocks")
}

func TestConverseNotesWhereEachReasoningBlockSat(t *testing.T) {
	t.Parallel()

	rec := &converseRecorder{responses: []string{converseReply(
		converseThought("plan", "s1"), converseToolUse("A"),
		converseThought("next", "s2"), converseEncrypted("opaque"), converseToolUse("B"),
	)}}
	choice := askConverse(t, rec.serve(t))

	assert.Equal(t, []reasoning.Block{
		{Text: "plan", Signature: []byte("s1")},
		{Text: "next", Signature: []byte("s2"), AfterToolCalls: 1},
		{Redacted: []byte("opaque"), AfterToolCalls: 1},
	}, choice.Reasoning.Sequence())
}

func TestTheConverseStreamKeepsEachBlockWithItsOwnSignature(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		enc := eventstream.NewEncoder()
		delta := func(index int, payload string) {
			writeConverseEvent(t, w, enc, "contentBlockDelta",
				fmt.Sprintf(`{"contentBlockIndex":%d,"delta":%s}`, index, payload))
		}
		stop := func(index int) {
			writeConverseEvent(t, w, enc, "contentBlockStop", fmt.Sprintf(`{"contentBlockIndex":%d}`, index))
		}

		writeConverseEvent(t, w, enc, "messageStart", `{"role":"assistant"}`)
		delta(0, `{"reasoningContent":{"text":"first "}}`)
		delta(0, `{"reasoningContent":{"text":"thought"}}`)
		delta(0, `{"reasoningContent":{"signature":"s1"}}`)
		stop(0)
		delta(1, `{"reasoningContent":{"text":"second"}}`)
		delta(1, `{"reasoningContent":{"signature":"s2"}}`)
		stop(1)
		writeConverseEvent(t, w, enc, "contentBlockStart",
			`{"contentBlockIndex":2,"start":{"toolUse":{"toolUseId":"A","name":"lookup"}}}`)
		delta(2, `{"toolUse":{"input":"{\"q\":\"A\"}"}}`)
		stop(2)
		delta(3, `{"reasoningContent":{"signature":"s3"}}`)
		stop(3)
		writeConverseEvent(t, w, enc, "messageStop", `{"stopReason":"max_tokens"}`)
		writeConverseEvent(t, w, enc, "metadata", `{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv, bedrock.WithModel(converseClaude), bedrock.WithConverseAPI())
	choice := askConverse(t, llm, llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

	assert.Equal(t, []reasoning.Block{
		{Text: "first thought", Signature: []byte("s1")},
		{Text: "second", Signature: []byte("s2")},
		{Signature: []byte("s3"), AfterToolCalls: 1},
	}, choice.Reasoning.Sequence())
}

func TestAReplayedConverseTurnGoesBackAsTheVendorSentIt(t *testing.T) {
	t.Parallel()

	for name, original := range map[string][]string{
		"reasoning and a progress update before one call": {
			converseThought("", "sig-reasoning"), converseThought("checking the index", "sig-update"), converseToolUse("A"),
		},
		"an update before each of two calls": {
			converseThought("plan", "s1"), converseToolUse("A"), converseThought("", "s2"), converseToolUse("B"),
		},
		"an update cut off after the last call": {
			converseThought("plan", "s1"), converseToolUse("A"), converseThought("", "s2"),
		},
		"an encrypted block between signed ones": {
			converseThought("a", "s1"), converseEncrypted("opaque"), converseThought("b", "s2"), converseToolUse("A"),
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			rec := &converseRecorder{responses: []string{
				converseReply(original...),
				`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn"}`,
			}}
			llm := rec.serve(t)
			choice := askConverse(t, llm)

			turn := []llms.ContentPart{llms.TextPartWithReasoning(choice.Content, choice.Reasoning)}
			results := make([]llms.ContentPart, 0, len(choice.ToolCalls))
			for _, call := range choice.ToolCalls {
				turn = append(turn, call)
				results = append(results, llms.ToolCallResponse{ToolCallID: call.ID, Name: "lookup", Content: "done"})
			}
			_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
				llms.TextParts(llms.ChatMessageTypeHuman, "look it up"),
				{Role: llms.ChatMessageTypeAI, Parts: turn},
				{Role: llms.ChatMessageTypeTool, Parts: results},
			}, llms.WithTools(lookupTools()))
			require.NoError(t, err)

			var sent []map[string]any
			require.NoError(t, json.Unmarshal([]byte("["+strings.Join(original, ",")+"]"), &sent))
			assert.Equal(t, converseShape(t, sent), rec.replayedAssistant(t, 1))
		})
	}
}

func TestAMergedConverseTurnKeepsTheReasoningOfEveryPart(t *testing.T) {
	t.Parallel()

	first := reasoning.FromBlocks([]reasoning.Block{{Text: "one", Signature: []byte("s1")}})
	second := reasoning.FromBlocks([]reasoning.Block{{Text: "two", Signature: []byte("s2")}})

	callA := llms.ToolCall{ID: "A", Type: "function", FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: `{"q":"A"}`}}
	both := []string{"thought one/s1", "thought two/s2", "text t1", "text t2", "tool A"}

	for name, tc := range map[string]struct {
		turn []llms.MessageContent
		want []string
	}{
		"two parts of one message": {turn: []llms.MessageContent{{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.TextPartWithReasoning("t1", first),
			llms.TextPartWithReasoning("t2", second),
			callA,
		}}}, want: both},
		"two consecutive messages": {turn: []llms.MessageContent{
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("t1", first)}},
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("t2", second), callA}},
		}, want: both},
		"a tool call ahead of the reasoning part": {turn: []llms.MessageContent{
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{callA, llms.TextPartWithReasoning("t1", first)}},
		}, want: []string{"thought one/s1", "text t1", "tool A"}},
		"an empty reasoning ahead of a real one": {turn: []llms.MessageContent{{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.TextPartWithReasoning("t1", &reasoning.ContentReasoning{}),
			llms.TextPartWithReasoning("t2", second),
			callA,
		}}}, want: []string{"thought two/s2", "text t1", "text t2", "tool A"}},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			rec := &converseRecorder{responses: []string{
				`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn"}`,
			}}
			history := append([]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "look it up")}, tc.turn...)
			history = append(history, llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
				llms.ToolCallResponse{ToolCallID: "A", Name: "lookup", Content: "done"},
			}})
			_, err := rec.serve(t).GenerateContent(context.Background(), history, llms.WithTools(lookupTools()))
			require.NoError(t, err)

			assert.Equal(t, tc.want, rec.replayedAssistant(t, 0))
		})
	}
}
