package bedrock_test

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

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type legacyRecorder struct {
	mu        sync.Mutex
	responses []string
	requests  [][]byte
}

func (l *legacyRecorder) serve(t *testing.T) *bedrock.LLM {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		l.mu.Lock()
		l.requests = append(l.requests, body)
		answer := l.responses[min(len(l.requests), len(l.responses))-1]
		l.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, answer)
	}))
	t.Cleanup(srv.Close)

	return bedrockLLMAgainst(t, srv, bedrock.WithModel(converseClaude))
}

func (l *legacyRecorder) replayedAssistant(t *testing.T, request int) []string {
	t.Helper()

	var payload struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(l.requests[request], &payload))
	for _, m := range payload.Messages {
		if m.Role == "assistant" {
			return legacyShape(m.Content)
		}
	}
	t.Fatalf("request %d carries no assistant turn", request)
	return nil
}

func legacyShape(content []map[string]any) []string {
	shape := make([]string, 0, len(content))
	for _, block := range content {
		switch block["type"] {
		case "thinking":
			shape = append(shape, fmt.Sprintf("thought %v/%v", block["thinking"], block["signature"]))
		case "redacted_thinking":
			shape = append(shape, fmt.Sprintf("encrypted %v", block["data"]))
		case "tool_use":
			shape = append(shape, fmt.Sprintf("tool %v", block["id"]))
		default:
			shape = append(shape, fmt.Sprintf("%v %v", block["type"], block["text"]))
		}
	}
	return shape
}

func legacyReply(blocks ...string) string {
	return `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[` + strings.Join(blocks, ",") +
		`],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`
}

func legacyThought(text, signature string) string {
	return fmt.Sprintf(`{"type":"thinking","thinking":%q,"signature":%q}`, text, signature)
}

func legacyToolUse(id string) string {
	return fmt.Sprintf(`{"type":"tool_use","id":%q,"name":"lookup","input":{"q":%q}}`, id, id)
}

func TestTheLegacyDoorKeepsEveryThinkingBlockWithItsSignature(t *testing.T) {
	t.Parallel()

	rec := &legacyRecorder{responses: []string{legacyReply(
		legacyThought("", "sig-reasoning"), legacyThought("checking the index", "sig-update"), legacyToolUse("A"),
	)}}
	choice := askConverse(t, rec.serve(t))

	assert.Equal(t, []reasoning.Block{
		{Signature: []byte("sig-reasoning")},
		{Text: "checking the index", Signature: []byte("sig-update")},
	}, choice.Reasoning.Sequence())
}

func TestTheLegacyDoorNotesWhereEachThinkingBlockSat(t *testing.T) {
	t.Parallel()

	rec := &legacyRecorder{responses: []string{legacyReply(
		legacyThought("plan", "s1"), legacyToolUse("A"),
		legacyThought("next", "s2"), `{"type":"redacted_thinking","data":"opaque"}`, legacyToolUse("B"),
	)}}
	choice := askConverse(t, rec.serve(t))

	assert.Equal(t, []reasoning.Block{
		{Text: "plan", Signature: []byte("s1")},
		{Text: "next", Signature: []byte("s2"), AfterToolCalls: 1},
		{Redacted: []byte("opaque"), AfterToolCalls: 1},
	}, choice.Reasoning.Sequence())
}

func TestTheLegacyStreamKeepsEachBlockWithItsOwnSignature(t *testing.T) {
	t.Parallel()

	chunks := []string{
		`{"type":"message_start","message":{"usage":{"input_tokens":1,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"first "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"thought"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"s1"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"thinking"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"second"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"signature_delta","signature":"s2"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"A","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"A\"}"}}`,
		`{"type":"content_block_stop","index":3}`,
		`{"type":"content_block_start","index":4,"content_block":{"type":"thinking"}}`,
		`{"type":"content_block_delta","index":4,"delta":{"type":"signature_delta","signature":"s3"}}`,
		`{"type":"content_block_stop","index":4}`,
		`{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"message":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
		`{"type":"message_stop"}`,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		enc := eventstream.NewEncoder()
		for _, chunk := range chunks {
			writeBedrockChunk(t, w, enc, chunk)
		}
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv, bedrock.WithModel(converseClaude))
	choice := askConverse(t, llm, llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

	assert.Equal(t, []reasoning.Block{
		{Text: "first thought", Signature: []byte("s1")},
		{Redacted: []byte("opaque")},
		{Text: "second", Signature: []byte("s2")},
		{Signature: []byte("s3"), AfterToolCalls: 1},
	}, choice.Reasoning.Sequence())
}

func TestAReplayedLegacyTurnGoesBackAsTheVendorSentIt(t *testing.T) {
	t.Parallel()

	for name, original := range map[string][]string{
		"reasoning and a progress update before one call": {
			legacyThought("", "sig-reasoning"), legacyThought("checking the index", "sig-update"),
			`{"type":"text","text":"working"}`, legacyToolUse("A"),
		},
		"an update before each of two calls": {
			legacyThought("plan", "s1"), `{"type":"text","text":"working"}`, legacyToolUse("A"),
			legacyThought("", "s2"), legacyToolUse("B"),
		},
		"an update cut off after the last call": {
			legacyThought("plan", "s1"), `{"type":"text","text":"working"}`, legacyToolUse("A"), legacyThought("", "s2"),
		},
		"a call with no text around it": {
			legacyThought("plan", "s1"), legacyToolUse("A"),
		},
		"an encrypted block between signed ones": {
			legacyThought("a", "s1"), `{"type":"redacted_thinking","data":"opaque"}`, legacyThought("b", "s2"),
			`{"type":"text","text":"working"}`, legacyToolUse("A"),
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			rec := &legacyRecorder{responses: []string{
				legacyReply(original...),
				`{"id":"msg_2","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],` +
					`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
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
			assert.Equal(t, legacyShape(sent), rec.replayedAssistant(t, 1))
		})
	}
}

func TestTheLegacyDoorOpensTheTurnWithItsReasoningWhateverThePartOrder(t *testing.T) {
	t.Parallel()

	rec := &legacyRecorder{responses: []string{
		`{"id":"msg_2","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],` +
			`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
	}}
	thought := reasoning.FromBlocks([]reasoning.Block{
		{Text: "plan", Signature: []byte("s1")},
		{Signature: []byte("s2"), AfterToolCalls: 1},
	})
	_, err := rec.serve(t).GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "look it up"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.ToolCall{ID: "A", Type: "function", FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: `{"q":"A"}`}},
			llms.TextPartWithReasoning("working", thought),
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "A", Name: "lookup", Content: "done"},
		}},
	}, llms.WithTools(lookupTools()))
	require.NoError(t, err)

	assert.Equal(t, []string{"thought plan/s1", "tool A", "thought /s2", "text working"}, rec.replayedAssistant(t, 0))
}

func TestAnEmptyCachedTextNeverReachesTheLegacyPayload(t *testing.T) {
	t.Parallel()

	thought := reasoning.FromBlocks([]reasoning.Block{{Text: "plan", Signature: []byte("s1")}})
	answer := `{"id":"msg_2","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`

	auto := &legacyRecorder{responses: []string{answer}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		auto.mu.Lock()
		auto.requests = append(auto.requests, body)
		auto.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, answer)
	}))
	t.Cleanup(srv.Close)
	llm := bedrockLLMAgainst(t, srv, bedrock.WithModel(converseClaude), bedrock.WithAutomaticCaching())
	_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "first"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("", thought)}},
		llms.TextParts(llms.ChatMessageTypeHuman, "second"),
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"thought plan/s1"}, auto.replayedAssistant(t, 0))
	assert.NotContains(t, string(auto.requests[0]), "cache_control",
		"a thinking block cannot carry the marker, so it goes with the empty text")

	explicit := &legacyRecorder{responses: []string{answer}}
	_, err = explicit.serve(t).GenerateContent(context.Background(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "look it up"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.ToolCall{ID: "A", Type: "function", FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: `{"q":"A"}`}},
			bedrock.WithCacheControl(llms.TextContent{}, bedrock.EphemeralCache()),
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "A", Name: "lookup", Content: "done"},
		}},
	}, llms.WithTools(lookupTools()))
	require.NoError(t, err)

	var payload struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(explicit.requests[0], &payload))
	assistant := payload.Messages[1].Content
	require.Len(t, assistant, 1)
	assert.Equal(t, "tool_use", assistant[0]["type"])
	assert.NotNil(t, assistant[0]["cache_control"], "the marker moves to the block before the empty text")
}

func TestTheMovedCacheMarkWalksPastThinkingAndKeepsItsTTL(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		last  reasoning.Block
		shape string
	}{
		{reasoning.Block{Text: "next", Signature: []byte("s2"), AfterToolCalls: 1}, "thought next/s2"},
		{reasoning.Block{Redacted: []byte("opaque"), AfterToolCalls: 1}, "encrypted opaque"},
	} {
		thought := reasoning.FromBlocks([]reasoning.Block{{Text: "plan", Signature: []byte("s1")}, tc.last})
		rec := &legacyRecorder{responses: []string{`{"id":"msg_2","type":"message","role":"assistant","model":"m",` +
			`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`}}
		_, err := rec.serve(t).GenerateContent(context.Background(), []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "look it up"),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.ToolCall{ID: "A", Type: "function", FunctionCall: &llms.FunctionCall{Name: "lookup", Arguments: `{"q":"A"}`}},
				bedrock.WithCacheControl(llms.TextPartWithReasoning("", thought), bedrock.EphemeralCacheOneHour()),
			}},
			{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
				llms.ToolCallResponse{ToolCallID: "A", Name: "lookup", Content: "done"},
			}},
		}, llms.WithTools(lookupTools()))
		require.NoError(t, err, tc.shape)

		var payload struct {
			Messages []struct {
				Content []map[string]any `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.Unmarshal(rec.requests[0], &payload))
		assistant := payload.Messages[1].Content
		require.Equal(t, []string{"thought plan/s1", "tool A", tc.shape}, legacyShape(assistant))
		assert.Equal(t, map[string]any{"type": "ephemeral", "ttl": "1h"}, assistant[1]["cache_control"], tc.shape)
		assert.Nil(t, assistant[2]["cache_control"], tc.shape)
	}
}
