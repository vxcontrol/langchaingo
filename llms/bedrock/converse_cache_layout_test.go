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
	"github.com/vxcontrol/langchaingo/llms/reasoning"
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

const growingConverseModel = "anthropic.claude-sonnet-4-5-20250929-v1:0"

type converseMark struct {
	after int
	kind  string
	ttl   string
}

type converseStep struct {
	parallel   int
	text       string
	afterCalls bool
}

func converseWideLoop(model string, steps int, step converseStep) []llms.MessageContent {
	chain := make([]llms.MessageContent, 0, 2+2*steps)
	chain = append(chain,
		llms.TextParts(llms.ChatMessageTypeSystem, "rules"),
		llms.TextParts(llms.ChatMessageTypeHuman, "scan the host"),
	)
	for i := range steps {
		block := reasoning.Block{Text: "plan", Signature: []byte("sig")}
		if step.afterCalls {
			block.AfterToolCalls = step.parallel
		}
		thought := reasoning.FromBlocks([]reasoning.Block{block}).WrittenBy(model)
		answer := llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.TextPartWithReasoning(step.text, thought),
		}}
		results := llms.MessageContent{Role: llms.ChatMessageTypeTool}
		for j := range step.parallel {
			id := fmt.Sprintf("call_%d_%d", i, j)
			answer.Parts = append(answer.Parts,
				llms.ToolCall{ID: id, Type: "function", FunctionCall: &llms.FunctionCall{Name: "nmap", Arguments: `{}`}})
			results.Parts = append(results.Parts, llms.ToolCallResponse{ToolCallID: id, Name: "nmap", Content: "open"})
		}
		chain = append(chain, answer, results)
	}
	return chain
}

func converseMarksAfterBlocks(t *testing.T, chain []llms.MessageContent, callOpts ...llms.CallOption) (
	[]string, []converseMark, *llms.ContentResponse,
) {
	t.Helper()

	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, converseAnswer)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv, bedrock.WithModel(growingConverseModel), bedrock.WithConverseAPI())
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
	var system []string
	for _, block := range sent.System {
		if point, ok := block["cachePoint"].(map[string]any); ok {
			system = append(system, fmt.Sprint(point["ttl"]))
		}
	}
	var marks []converseMark
	blocks, kind := 0, ""
	for _, msg := range sent.Messages {
		for _, block := range msg.Content {
			if point, ok := block["cachePoint"].(map[string]any); ok {
				marks = append(marks, converseMark{blocks - 1, kind, fmt.Sprint(point["ttl"])})
				continue
			}
			for key := range block {
				kind = key
			}
			blocks++
		}
	}
	return system, marks, resp
}

func TestAGrowingConverseHistoryChainsItsHourLongWritesAcrossWideSteps(t *testing.T) {
	t.Parallel()

	for _, step := range []converseStep{
		{1, "checking", false}, {2, "", false}, {3, "", false}, {3, "checking", false}, {6, "checking", false},
		{8, "checking", false}, {10, "checking", false}, {2, "", true}, {3, "checking", true},
	} {
		parallel, previous := step.parallel, -1
		width := 1 + 2*parallel
		if step.text != "" {
			width++
		}
		for steps := range 30 {
			system, marks, _ := converseMarksAfterBlocks(t, converseWideLoop(growingConverseModel, steps, step),
				llms.WithCacheLayout(llms.CacheLayoutGrowing))
			require.LessOrEqual(t, len(system)+len(marks), 4, "%d calls a step, %d steps", parallel, steps)
			furthest, reached := -1, previous < 0
			for i, mark := range marks {
				require.NotEqual(t, "reasoningContent", mark.kind, "%d calls a step, %d steps: a point after reasoning", parallel, steps)
				if i < len(marks)-1 {
					require.Equal(t, "1h", mark.ttl, "%d calls a step, %d steps: an hour-long point before a five-minute one", parallel, steps)
				}
				if mark.ttl == "1h" {
					furthest = max(furthest, mark.after)
					reached = reached || mark.after >= previous && mark.after-previous < 20
				}
			}
			require.True(t, reached, "%d calls a step, %d steps: no hour-long point reads the write at block %d", parallel, steps, previous)
			if width <= 18 {
				require.Less(t, marks[len(marks)-1].after-furthest, 20,
					"%d calls a step, %d steps: the hour-long write keeps up with a step of %d blocks", parallel, steps, width)
			}
			previous = furthest
		}
	}
}

func TestAGrowingConverseHistoryMovesItsHourLongPointEveryTwelveBlocks(t *testing.T) {
	t.Parallel()

	_, marks, _ := converseMarksAfterBlocks(t, converseLoop(11), llms.WithCacheLayout(llms.CacheLayoutGrowing))
	require.Equal(t, []converseMark{{0, "text", "1h"}, {12, "toolResult", "1h"}, {22, "toolResult", "5m"}}, marks)
}

func TestAGrowingConverseHistoryPlacesThePointsInPlaceOfTheCallersOwn(t *testing.T) {
	t.Parallel()

	chain := converseWideLoop(growingConverseModel, 14, converseStep{1, "checking", false})
	chain[1] = llms.MessageContent{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{
		bedrock.WithCacheControl(llms.TextPart("the report"), bedrock.EphemeralCache()),
	}}

	system, marks, resp := converseMarksAfterBlocks(t, chain, llms.WithCacheLayout(llms.CacheLayoutGrowing))
	require.LessOrEqual(t, len(system)+len(marks), 4)
	for _, mark := range marks[:len(marks)-1] {
		require.Equal(t, "1h", mark.ttl, "an hour-long point before a five-minute one")
	}
	var dropped []llms.Warning
	for _, w := range resp.Warnings {
		if w.Option == "WithCacheControl" {
			dropped = append(dropped, w)
		}
	}
	require.Len(t, dropped, 1)
	require.Equal(t, llms.WarningDrop, dropped[0].Kind)
	require.Equal(t, "1 marker", dropped[0].Asked)
}

func invokeModelRequest(t *testing.T, clientOpts []bedrock.Option, chain []llms.MessageContent, opts ...llms.CallOption) (
	[]byte, *llms.ContentResponse,
) {
	t.Helper()

	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, legacyAnswer)
	}))
	t.Cleanup(srv.Close)
	resp, err := bedrockLLMAgainst(t, srv, append([]bedrock.Option{bedrock.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0")}, clientOpts...)...).
		GenerateContent(t.Context(), chain, opts...)
	require.NoError(t, err)
	return raw, resp
}

func TestTheInvokeModelRequestHonoursNoLayoutAndReportsAGrowingOne(t *testing.T) {
	t.Parallel()

	auto := []bedrock.Option{bedrock.WithAutomaticCaching()}
	raw, _ := invokeModelRequest(t, auto, converseLoop(3))
	require.NotEmpty(t, collectJSONObjects(t, raw, "cache_control"), "the door's automatic caching as before")

	raw, resp := invokeModelRequest(t, auto, converseLoop(3), llms.WithCacheLayout(llms.CacheLayoutNone))
	require.Empty(t, collectJSONObjects(t, raw, "cache_control"))
	require.Empty(t, resp.Warnings)

	raw, resp = invokeModelRequest(t, auto, converseLoop(3), llms.WithCacheLayout(llms.CacheLayoutGrowing))
	require.NotEmpty(t, collectJSONObjects(t, raw, "cache_control"), "the door's own markers")
	layout := layoutWarnings(resp)
	require.Len(t, layout, 1)
	require.Equal(t, llms.WarningSubstitute, layout[0].Kind)
	require.NotEmpty(t, layout[0].Sent)

	raw, resp = invokeModelRequest(t, nil, converseLoop(3), llms.WithCacheLayout(llms.CacheLayoutGrowing))
	require.Empty(t, collectJSONObjects(t, raw, "cache_control"))
	layout = layoutWarnings(resp)
	require.Len(t, layout, 1, "without automatic caching nothing reaches the wire")
	require.Equal(t, llms.WarningDrop, layout[0].Kind)
	require.Empty(t, layout[0].Sent)

	cohere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"generations":[{"text":"ok","finish_reason":"COMPLETE"}]}`)
	}))
	t.Cleanup(cohere.Close)
	resp, err := bedrockLLMAgainst(t, cohere, bedrock.WithModel("cohere.command-text-v14")).
		GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithCacheLayout(llms.CacheLayoutGrowing))
	require.NoError(t, err)
	require.Empty(t, layoutWarnings(resp), "a model the door does not cache has no markers to place")
}

func TestAFirstInvokeModelRequestReportsAGrowingLayoutAsDropped(t *testing.T) {
	t.Parallel()

	raw, resp := invokeModelRequest(t, []bedrock.Option{bedrock.WithAutomaticCaching()}, converseLoop(0),
		llms.WithCacheLayout(llms.CacheLayoutGrowing))
	require.Empty(t, collectJSONObjects(t, raw, "cache_control"), "no answer for the door to mark")
	layout := layoutWarnings(resp)
	require.Len(t, layout, 1)
	require.Equal(t, llms.WarningDrop, layout[0].Kind)
	require.Empty(t, layout[0].Sent)
}

func layoutWarnings(resp *llms.ContentResponse) []llms.Warning {
	var layout []llms.Warning
	for _, w := range resp.Warnings {
		if w.Option == "WithCacheLayout" {
			layout = append(layout, w)
		}
	}
	return layout
}

func TestAGrowingConverseHistoryPointsAtTheStartOfTheLatestTurn(t *testing.T) {
	t.Parallel()

	chain := append(converseLoop(6),
		llms.TextParts(llms.ChatMessageTypeAI, "done"),
		llms.TextParts(llms.ChatMessageTypeHuman, "now the web server"))
	chain = append(chain,
		llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{
			llms.ToolCall{ID: "call_web", Type: "function", FunctionCall: &llms.FunctionCall{Name: "nmap", Arguments: `{}`}},
		}},
		llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{
			llms.ToolCallResponse{ToolCallID: "call_web", Name: "nmap", Content: "open"},
		}})

	_, marks, _ := converseMarksAfterBlocks(t, chain, llms.WithCacheLayout(llms.CacheLayoutGrowing))
	require.Contains(t, marks, converseMark{14, "text", "1h"}, "the second human message")
}
