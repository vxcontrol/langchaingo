package anthropic_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type cacheMark struct {
	position int
	ttl      string
}

type sentCacheMarks struct {
	system   []string
	messages []cacheMark
	lastAt   int
}

type loopShape struct {
	parallel                   int
	thinking, afterCalls, text bool
}

func agentLoop(calls int, shape loopShape) []llms.MessageContent {
	chain := make([]llms.MessageContent, 0, 2+2*calls)
	chain = append(chain,
		llms.TextParts(llms.ChatMessageTypeSystem, "rules"),
		llms.TextParts(llms.ChatMessageTypeHuman, "scan the host"),
	)
	for i := range calls {
		answer := llms.MessageContent{Role: llms.ChatMessageTypeAI}
		parallel := max(1, shape.parallel)
		if shape.thinking {
			block := reasoning.Block{Text: "plan", Signature: []byte("sig")}
			if shape.afterCalls {
				block.AfterToolCalls = parallel
			}
			thought := reasoning.FromBlocks([]reasoning.Block{block}).WrittenBy("claude-opus-5-5")
			text := ""
			if shape.text {
				text = "checking the ports"
			}
			answer.Parts = append(answer.Parts, llms.TextPartWithReasoning(text, thought))
		}
		results := llms.MessageContent{Role: llms.ChatMessageTypeTool}
		for j := range parallel {
			id := fmt.Sprintf("call_%d_%d", i, j)
			answer.Parts = append(answer.Parts,
				llms.ToolCall{ID: id, Type: "function", FunctionCall: &llms.FunctionCall{Name: "nmap", Arguments: `{}`}})
			results.Parts = append(results.Parts, llms.ToolCallResponse{ToolCallID: id, Name: "nmap", Content: "open"})
		}
		chain = append(chain, answer, results)
	}
	return chain
}

func growingCacheMarks(t *testing.T, chain []llms.MessageContent, opts ...llms.CallOption) sentCacheMarks {
	t.Helper()

	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"m","type":"message","role":"assistant","model":"claude-opus-5-5",`+
			`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := anthropic.New(anthropic.WithToken("test-key"), anthropic.WithBaseURL(srv.URL), anthropic.WithModel("claude-opus-5-5"),
		anthropic.WithDefaultCacheStrategy(anthropic.CacheStrategy{CacheTools: true, CacheSystem: true, CacheMessages: true}))
	require.NoError(t, err)
	_, err = llm.GenerateContent(t.Context(), chain, append([]llms.CallOption{
		llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
			Name: "nmap", Parameters: map[string]any{"type": "object"},
		}}}),
	}, opts...)...)
	require.NoError(t, err)

	var sent struct {
		Tools    []map[string]any `json:"tools"`
		System   json.RawMessage  `json:"system"`
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(body, &sent))
	var marks sentCacheMarks
	for _, tool := range sent.Tools {
		if mark, ok := tool["cache_control"].(map[string]any); ok {
			marks.system = append(marks.system, "tool:"+fmt.Sprint(mark["ttl"]))
		}
	}
	var system []map[string]any
	if json.Unmarshal(sent.System, &system) == nil {
		for _, block := range system {
			if mark, ok := block["cache_control"].(map[string]any); ok {
				marks.system = append(marks.system, fmt.Sprint(mark["ttl"]))
			}
		}
	}
	position := -1
	for _, msg := range sent.Messages {
		previous := ""
		for _, block := range msg.Content {
			kind, _ := block["type"].(string)
			if kind != previous || (kind != "tool_use" && kind != "tool_result") {
				position++
			}
			previous = kind
			if mark, ok := block["cache_control"].(map[string]any); ok {
				marks.messages = append(marks.messages, cacheMark{position, fmt.Sprint(mark["ttl"])})
			}
		}
	}
	marks.lastAt = position
	return marks
}

func TestAGrowingHistoryKeepsAnHourLongWriteInReachOfTheNextRequest(t *testing.T) {
	t.Parallel()

	for _, shape := range []loopShape{
		{}, {parallel: 3}, {thinking: true}, {thinking: true, text: true},
		{parallel: 3, thinking: true}, {parallel: 3, thinking: true, afterCalls: true},
	} {
		growingRequestsKeepTheirWritesInReach(t, shape)
	}
}

func growingRequestsKeepTheirWritesInReach(t *testing.T, shape loopShape) {
	t.Helper()

	previous := -1
	for calls := range 40 {
		marks := growingCacheMarks(t, agentLoop(calls, shape), llms.WithCacheLayout(llms.CacheLayoutGrowing))

		require.Equal(t, []string{"1h"}, marks.system, "%d calls: the system prompt, which also covers the tools", calls)
		require.LessOrEqual(t, len(marks.system)+len(marks.messages), 4, "%d calls", calls)
		require.Equal(t, cacheMark{0, "1h"}, marks.messages[0], "%d calls: the start of the turn", calls)
		last := marks.messages[len(marks.messages)-1]
		require.Equal(t, marks.lastAt, last.position, "%d calls: the last block", calls)
		hourLong, hours, minutes := -1, 0, 0
		for i, mark := range marks.messages {
			if i < len(marks.messages)-1 {
				require.Equal(t, "1h", mark.ttl, "%d calls: an hour-long write before a five-minute one", calls)
			}
			if mark.ttl == "1h" {
				hourLong = mark.position
				hours++
			} else {
				minutes++
			}
		}
		require.LessOrEqual(t, hours, 2, "%d calls: the start of the turn and the moving marker", calls)
		require.Less(t, marks.lastAt-hourLong, 20, "%d calls: an hour-long write follows the history", calls)
		require.LessOrEqual(t, minutes, 1, "%d calls: the last block", calls)
		if previous >= 0 {
			reached := false
			for _, mark := range marks.messages {
				reached = reached || (mark.position >= previous && mark.position-previous < 20)
			}
			require.True(t, reached, "%d calls: no marker looks back far enough to read the write at %d", calls, previous)
		}
		previous = hourLong
	}
}

func TestNoLayoutPlacesNoMarkers(t *testing.T) {
	t.Parallel()

	marks := growingCacheMarks(t, agentLoop(3, loopShape{}), llms.WithCacheLayout(llms.CacheLayoutNone))
	require.Empty(t, marks.system)
	require.Empty(t, marks.messages)
}
