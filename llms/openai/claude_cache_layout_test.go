package openai

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

type cacheMark struct {
	at  string
	ttl any
}

func cacheMarksSent(t *testing.T, body map[string]any) []cacheMark {
	t.Helper()

	var marks []cacheMark
	mark := func(at string, control any) {
		if control == nil {
			return
		}
		fields, ok := control.(map[string]any)
		require.True(t, ok, "%s: %v", at, control)
		require.Equal(t, "ephemeral", fields["type"], at)
		marks = append(marks, cacheMark{at, fields["ttl"]})
	}
	tools, _ := body["tools"].([]any)
	for i, tool := range tools {
		function, _ := tool.(map[string]any)["function"].(map[string]any)
		mark(fmt.Sprintf("tool %d", i), function["cache_control"])
	}
	messages, _ := body["messages"].([]any)
	for i, raw := range messages {
		msg, _ := raw.(map[string]any)
		parts, _ := msg["content"].([]any)
		for j, part := range parts {
			mark(fmt.Sprintf("%s %d part %d", msg["role"], i, j), part.(map[string]any)["cache_control"])
		}
	}
	raw, err := json.Marshal(body)
	require.NoError(t, err)
	require.Equal(t, len(marks), strings.Count(string(raw), "cache_control"), "a marker outside a text part or a tool")
	return marks
}

func nmapCall(id string) []llms.MessageContent {
	return []llms.MessageContent{
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.ToolCall{ID: id, Type: "function", FunctionCall: &llms.FunctionCall{Name: "nmap", Arguments: `{}`}},
		}},
		{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: id, Name: "nmap", Content: "open"},
		}},
	}
}

func TestAGrowingHistoryMarksClaudeBehindAGatewayWhereLiteLLMTakesTheMarker(t *testing.T) {
	t.Parallel()

	hour := "1h"
	system := llms.TextParts(llms.ChatMessageTypeSystem, "rules", "the scope")
	task := llms.MessageContent{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{
		llms.TextPart("scan the host"), llms.ImageURLContent{URL: "https://example.com/net.png"},
	}}
	loop := append(append([]llms.MessageContent{system, task}, nmapCall("call_0")...), nmapCall("call_1")...)
	nextTurn := append(append([]llms.MessageContent{}, loop...), llms.TextParts(llms.ChatMessageTypeAI, "done"),
		llms.TextParts(llms.ChatMessageTypeHuman, "now the web server"))
	noSystem := append([]llms.MessageContent{task}, nmapCall("call_0")...)
	oneSystemPart := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeSystem, "rules"), task}
	tools := llms.WithTools([]llms.Tool{
		{Type: "function", Function: &llms.FunctionDefinition{Name: "curl", Parameters: map[string]any{"type": "object"}}},
		{Type: "function", Function: &llms.FunctionDefinition{Name: "nmap", Parameters: map[string]any{"type": "object"}}},
	})

	for name, tc := range map[string]struct {
		baseURL, model string
		chain          []llms.MessageContent
		want           []cacheMark
	}{
		"system and the start of the turn": {gatewayBaseURL, "anthropic/claude-sonnet-5-5", loop,
			[]cacheMark{{"system 0 part 1", hour}, {"user 1 part 0", hour}}},
		"a later turn": {gatewayBaseURL, "anthropic/claude-sonnet-5-5", nextTurn,
			[]cacheMark{{"system 0 part 1", hour}, {"user 7 part 0", hour}}},
		"no system marks the last tool": {gatewayBaseURL, "anthropic/claude-sonnet-5-5", noSystem,
			[]cacheMark{{"tool 1", hour}, {"user 0 part 0", hour}}},
		"a one-part system": {gatewayBaseURL, "anthropic/claude-sonnet-5-5", oneSystemPart,
			[]cacheMark{{"system 0 part 0", hour}, {"user 1 part 0", hour}}},
		"a gateway alias": {gatewayBaseURL, "claude-sonnet-5-5", loop, []cacheMark{{"system 0 part 1", hour}, {"user 1 part 0", hour}}},
		"bedrock for an hour": {gatewayBaseURL, "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", loop,
			[]cacheMark{{"system 0 part 1", hour}, {"user 1 part 0", hour}}},
		"bedrock for five minutes": {gatewayBaseURL, "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0", loop,
			[]cacheMark{{"system 0 part 1", nil}, {"user 1 part 0", nil}}},
		"vertex":     {gatewayBaseURL, "vertex_ai/claude-sonnet-5-5", loop, nil},
		"openrouter": {openRouterBaseURL, "anthropic/claude-sonnet-5-5", loop, nil},
		"gemini":     {gatewayBaseURL, "gemini/gemini-3.5-pro", loop, nil},
		"gpt":        {gatewayBaseURL, "openai/gpt-5.5", loop, nil},
		"gpt alias":  {gatewayBaseURL, "gpt-5.5", loop, nil},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			sent := sentFor(t, tc.baseURL, tc.model, tc.chain, tools, llms.WithCacheLayout(llms.CacheLayoutGrowing))
			require.Empty(t, sent.err)
			require.Equal(t, tc.want, cacheMarksSent(t, sent.body))
			require.IsType(t, llms.TextContent{}, system.Parts[1])
			require.IsType(t, llms.TextContent{}, task.Parts[0])
		})
	}
}

func TestClaudeBehindAGatewayGetsNoMarkersWithoutAGrowingHistory(t *testing.T) {
	t.Parallel()

	chain := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "rules"), llms.TextParts(llms.ChatMessageTypeHuman, "scan the host"),
	}
	for _, opts := range [][]llms.CallOption{nil, {llms.WithCacheLayout(llms.CacheLayoutNone)}} {
		sent := sentFor(t, gatewayBaseURL, "anthropic/claude-sonnet-5-5", chain, opts...)
		require.Empty(t, sent.err)
		require.Empty(t, cacheMarksSent(t, sent.body))
		require.Equal(t, "rules", sent.body["messages"].([]any)[0].(map[string]any)["content"])
	}
}
