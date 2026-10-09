package bedrock_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

type conversePoint struct {
	block int
	ttl   string
}

type converseCacheMarks struct {
	system []string
	points []conversePoint
	blocks int
}

func converseLoop(calls int) []llms.MessageContent {
	chain := make([]llms.MessageContent, 0, 2+2*calls)
	chain = append(chain,
		llms.TextParts(llms.ChatMessageTypeSystem, "rules"),
		llms.TextParts(llms.ChatMessageTypeHuman, "scan the host"),
	)
	for i := range calls {
		id := fmt.Sprintf("call_%d", i)
		chain = append(chain,
			llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
				llms.ToolCall{ID: id, Type: "function", FunctionCall: &llms.FunctionCall{Name: "nmap", Arguments: `{}`}},
			}},
			llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
				llms.ToolCallResponse{ToolCallID: id, Name: "nmap", Content: "open"},
			}},
		)
	}
	return chain
}

func converseCacheMarksFor(t *testing.T, model string, chain []llms.MessageContent, clientOpts []bedrock.Option,
	callOpts ...llms.CallOption,
) (converseCacheMarks, *llms.ContentResponse) {
	t.Helper()

	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, converseAnswer)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv, append([]bedrock.Option{bedrock.WithModel(model), bedrock.WithConverseAPI()}, clientOpts...)...)
	resp, err := llm.GenerateContent(context.Background(), chain, append([]llms.CallOption{
		llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
			Name: "nmap", Parameters: map[string]any{"type": "object"},
		}}}),
	}, callOpts...)...)
	require.NoError(t, err)

	var sent struct {
		System   []map[string]any `json:"system"`
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(body, &sent))
	var marks converseCacheMarks
	for _, block := range sent.System {
		if point, ok := block["cachePoint"].(map[string]any); ok {
			marks.system = append(marks.system, fmt.Sprint(point["ttl"]))
		}
	}
	for _, msg := range sent.Messages {
		for _, block := range msg.Content {
			if point, ok := block["cachePoint"].(map[string]any); ok {
				marks.points = append(marks.points, conversePoint{marks.blocks - 1, fmt.Sprint(point["ttl"])})
			}
			marks.blocks++
		}
	}
	return marks, resp
}

func TestAGrowingConverseHistoryKeepsAnHourLongWriteInReach(t *testing.T) {
	t.Parallel()

	const model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	for calls := range 40 {
		marks, _ := converseCacheMarksFor(t, model, converseLoop(calls), nil, llms.WithCacheLayout(llms.CacheLayoutGrowing))

		require.Equal(t, []string{"1h"}, marks.system, "%d calls", calls)
		require.LessOrEqual(t, len(marks.system)+len(marks.points), 4, "%d calls", calls)
		require.Equal(t, conversePoint{0, "1h"}, marks.points[0], "%d calls: the start of the turn", calls)
		last := marks.points[len(marks.points)-1]
		require.Equal(t, marks.blocks-2, last.block, "%d calls: a point after the last block", calls)
		hourLong, hours, minutes := -1, 0, 0
		for i, point := range marks.points {
			if i < len(marks.points)-1 {
				require.Equal(t, "1h", point.ttl, "%d calls: an hour-long point before a five-minute one", calls)
			}
			if point.ttl == "1h" {
				hourLong = point.block
				hours++
			} else {
				minutes++
			}
		}
		require.LessOrEqual(t, hours, 2, "%d calls", calls)
		require.LessOrEqual(t, minutes, 1, "%d calls", calls)
		require.Less(t, marks.blocks-1-hourLong, 20, "%d calls: an hour-long write follows the history", calls)
	}
}

func TestAGrowingConverseHistoryOnAFiveMinuteModelAsksNoHour(t *testing.T) {
	t.Parallel()

	marks, resp := converseCacheMarksFor(t, "anthropic.claude-opus-4-1-20250805-v1:0", converseLoop(3), nil,
		llms.WithCacheLayout(llms.CacheLayoutGrowing))
	require.NotEmpty(t, marks.points)
	for _, point := range append(marks.points, conversePoint{ttl: marks.system[0]}) {
		require.Equal(t, "<nil>", point.ttl, "AWS takes no ttl from a model it caches for five minutes only")
	}
	require.Empty(t, resp.Warnings)
}

func TestNoConverseLayoutPlacesNoPoints(t *testing.T) {
	t.Parallel()

	const model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	auto := []bedrock.Option{bedrock.WithAutomaticCaching()}

	marks, _ := converseCacheMarksFor(t, model, converseLoop(3), auto)
	require.NotEmpty(t, marks.points, "without a layout the door's automatic caching applies as before")

	marks, _ = converseCacheMarksFor(t, model, converseLoop(3), auto, llms.WithCacheLayout(llms.CacheLayoutNone))
	require.Empty(t, marks.system)
	require.Empty(t, marks.points)

	answered := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "rules"),
		llms.TextParts(llms.ChatMessageTypeHuman, "hi"),
		llms.TextParts(llms.ChatMessageTypeAI, "hello"),
		llms.TextParts(llms.ChatMessageTypeHuman, "go on"),
	}
	marks, _ = converseCacheMarksFor(t, model, answered, auto, llms.WithCacheLayout(llms.CacheLayoutNone))
	require.Empty(t, marks.system)
	require.Empty(t, marks.points, "the answer the door would mark")

	explicit := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "rules"),
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{bedrock.WithCacheControl(llms.TextPart("the report"), bedrock.EphemeralCache())}},
	}
	marks, _ = converseCacheMarksFor(t, model, explicit, nil, llms.WithCacheLayout(llms.CacheLayoutNone))
	require.Empty(t, marks.system)
	require.Len(t, marks.points, 1, "the caller's own point")

	marks, _ = converseCacheMarksFor(t, "us.deepseek.r1-v1:0", converseLoop(3), nil, llms.WithCacheLayout(llms.CacheLayoutGrowing))
	require.Empty(t, marks.system, "a model Bedrock does not cache")
	require.Empty(t, marks.points)
}
