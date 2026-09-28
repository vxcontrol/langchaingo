package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
	"github.com/vxcontrol/langchaingo/llms/structuredoutput"
)

// fallbackReply is what the stub vendor answers: one choice, as a single
// completion or, for a streaming request, as SSE chunks of the given pieces.
// A streamed piece can carry a reasoning delta too, as MiniMax's do.
type fallbackReply struct {
	pieces       []string
	reasonings   []string
	finishReason string
	toolCall     bool
}

// fallbackServer answers every chat request with the same reply and keeps the
// request bodies for the test goroutine to decode.
type fallbackServer struct {
	*httptest.Server

	mu     sync.Mutex
	bodies [][]byte
}

func newFallbackServer(t *testing.T, reply fallbackReply) *fallbackServer {
	t.Helper()

	if reply.finishReason == "" {
		reply.finishReason = "stop"
	}
	fs := &fallbackServer{}
	fs.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		fs.mu.Lock()
		fs.bodies = append(fs.bodies, body)
		fs.mu.Unlock()

		content, _ := json.Marshal(strings.Join(reply.pieces, ""))
		toolCalls := ""
		if reply.toolCall {
			toolCalls = `,"tool_calls":[{"id":"call_1","type":"function",` +
				`"function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]`
		}
		if !strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"x","object":"chat.completion","created":1,"model":"m",`+
				`"choices":[{"index":0,"message":{"role":"assistant","content":`+string(content)+toolCalls+`},`+
				`"finish_reason":"`+reply.finishReason+`"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for i, piece := range reply.pieces {
			delta, _ := json.Marshal(piece)
			thought := ""
			if i < len(reply.reasonings) && reply.reasonings[i] != "" {
				raw, _ := json.Marshal(reply.reasonings[i])
				thought = `,"reasoning":` + string(raw)
			}
			_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m",`+
				`"choices":[{"index":0,"delta":{"role":"assistant","content":`+string(delta)+thought+`},"finish_reason":null}]}`+"\n\n")
		}
		_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m",`+
			`"choices":[{"index":0,"delta":{},"finish_reason":"`+reply.finishReason+`"}]}`+"\n\ndata: [DONE]\n\n")
	}))
	t.Cleanup(fs.Close)
	return fs
}

func (fs *fallbackServer) requests() int {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	return len(fs.bodies)
}

// wireRequest is the part of a chat request these tests read.
type wireRequest struct {
	ResponseFormat json.RawMessage `json:"response_format"`
	Messages       []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"messages"`
	Tools []any `json:"tools"`
}

func (fs *fallbackServer) request(t *testing.T, i int) wireRequest {
	t.Helper()

	fs.mu.Lock()
	defer fs.mu.Unlock()
	require.Greater(t, len(fs.bodies), i, "the server got fewer requests than expected")
	var req wireRequest
	require.NoError(t, json.Unmarshal(fs.bodies[i], &req))
	return req
}

// messageText returns a wire content as text: the string itself, or the text
// parts of a content array joined, and whether it was an array.
func messageText(t *testing.T, content json.RawMessage) (string, bool) {
	t.Helper()

	var text string
	if json.Unmarshal(content, &text) == nil {
		return text, false
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	require.NoError(t, json.Unmarshal(content, &parts))
	var texts []string
	for _, part := range parts {
		if part.Type == "text" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "|"), true
}

func newFallbackLLM(t *testing.T, serverURL, model string, extra ...Option) *LLM {
	t.Helper()

	opts := append([]Option{WithBaseURL(serverURL), WithToken("test"), WithModel(model), WithStructuredOutputFallback()}, extra...)
	llm, err := New(opts...)
	require.NoError(t, err)
	return llm
}

func answerSchema() llms.CallOption {
	return llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "answer", Schema: objectSchema()})
}

func TestTheFallbackSendsJSONObjectAndTheSchemaInThePrompt(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"deepseek-flash", "deepseek-v4-pro", "deepseek/deepseek-v4-pro", "glm-5.3", "zai/glm-5.3"} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			srv := newFallbackServer(t, fallbackReply{pieces: []string{`{"answer":"42"}`}})
			resp, err := newFallbackLLM(t, srv.URL, model).GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}, answerSchema())
			require.NoError(t, err)
			assert.Equal(t, `{"answer":"42"}`, resp.Choices[0].Content)

			req := srv.request(t, 0)
			assert.JSONEq(t, `{"type":"json_object"}`, string(req.ResponseFormat), "no json_schema reaches the vendor")
			text, array := messageText(t, req.Messages[0].Content)
			assert.False(t, array, "a text message stays a string")
			assert.Equal(t, "question\n\n"+structuredoutput.PromptInstruction(objectSchema(), false), text)

			var substitute llms.Warning
			for _, w := range resp.Warnings {
				if w.Option == "WithStructuredOutput" {
					substitute = w
				}
			}
			assert.Equal(t, llms.WarningSubstitute, substitute.Kind)
			assert.Equal(t, model, substitute.Model)
			assert.Equal(t, "answer", substitute.Asked)
			assert.Equal(t, "json_object and a prompt instruction", substitute.Sent)
			assert.Equal(t, "the vendor's chat completions response_format takes only text and json_object, "+
				"so the schema travels in the prompt and the answer is validated locally", substitute.Reason)
		})
	}
}

func TestTheFallbackSendsMiniMaxOnlyTheSchemaInThePrompt(t *testing.T) {
	t.Parallel()

	for _, model := range []string{"MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M3", "minimax/MiniMax-M3"} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			// MiniMax puts its thinking at the head of content; the client splits it off.
			srv := newFallbackServer(t, fallbackReply{pieces: []string{"<think>\nThe user wants JSON.\n</think>\n\n", `{"answer":"42"}`}})
			resp, err := newFallbackLLM(t, srv.URL, model).GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}, answerSchema(), llms.WithJSONMode())
			require.NoError(t, err)
			assert.Equal(t, `{"answer":"42"}`, resp.Choices[0].Content)
			require.NotNil(t, resp.Choices[0].Reasoning)
			assert.Equal(t, "The user wants JSON.", resp.Choices[0].Reasoning.Content)

			req := srv.request(t, 0)
			assert.Empty(t, req.ResponseFormat, "MiniMax's API has no response_format, not even json_object")
			text, _ := messageText(t, req.Messages[0].Content)
			assert.Equal(t, "question\n\n"+structuredoutput.PromptInstruction(objectSchema(), false), text)

			var warnings []llms.Warning
			for _, w := range resp.Warnings {
				if w.Option == "WithStructuredOutput" || w.Option == "WithJSONMode" {
					warnings = append(warnings, w)
				}
			}
			require.Len(t, warnings, 1, "JSON mode under the schema is not reported on its own")
			assert.Equal(t, llms.WarningSubstitute, warnings[0].Kind)
			assert.Equal(t, "WithStructuredOutput", warnings[0].Option)
			assert.Equal(t, "a prompt instruction", warnings[0].Sent)
			assert.Equal(t, "the vendor's chat completions API takes no response_format for this model, "+
				"so the schema travels in the prompt and the answer is validated locally", warnings[0].Reason)
		})
	}
}

// placementCase is a conversation and where the fallback must put the schema.
type placementCase struct {
	name       string
	messages   []llms.MessageContent
	opts       []llms.CallOption
	roles      []string
	instructed int
	prefix     string
	array      bool
	toolsNote  bool
}

func placementCases() []placementCase {
	toolCall := llms.ToolCall{
		ID: "call_1", Type: "function",
		FunctionCall: &llms.FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`},
	}
	return []placementCase{
		{
			name: "only the last of several user turns",
			messages: []llms.MessageContent{
				llms.TextParts(llms.ChatMessageTypeSystem, "sys"),
				llms.TextParts(llms.ChatMessageTypeHuman, "first"),
				llms.TextParts(llms.ChatMessageTypeAI, "reply"),
				llms.TextParts(llms.ChatMessageTypeHuman, "second"),
			},
			roles: []string{"system", "user", "assistant", "user"}, instructed: 3, prefix: "second\n\n",
		},
		{
			name: "the user turn before a tool round, telling the model it may still call tools",
			messages: []llms.MessageContent{
				llms.TextParts(llms.ChatMessageTypeHuman, "weather in Paris?"),
				{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{toolCall}},
				{Role: llms.ChatMessageTypeTool, Parts: []llms.ContentPart{llms.ToolCallResponse{
					ToolCallID: toolCall.ID, Name: "get_weather", Content: "sunny",
				}}},
			},
			opts:  []llms.CallOption{llms.WithTools([]llms.Tool{weatherTool()})},
			roles: []string{"user", "assistant", "tool"}, instructed: 0, prefix: "weather in Paris?\n\n", toolsNote: true,
		},
		{
			name:     "a user turn of its own when there is none, since Z.ai answers no conversation without one",
			messages: []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeSystem, "sys")},
			roles:    []string{"system", "user"}, instructed: 1, prefix: structuredoutput.SchemaInstruction,
		},
		{
			name: "a text part of its own after an image",
			messages: []llms.MessageContent{{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{
				llms.TextContent{Text: "what is this?"},
				llms.ImageURLContent{URL: "https://example.com/cat.png"},
			}}},
			roles: []string{"user"}, instructed: 0, prefix: "what is this?|" + structuredoutput.SchemaInstruction, array: true,
		},
	}
}

func TestTheFallbackCarriesTheSchemaInThePrompt(t *testing.T) {
	t.Parallel()

	for _, tc := range placementCases() {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			srv := newFallbackServer(t, fallbackReply{pieces: []string{`{"answer":"42"}`}})
			llm := newFallbackLLM(t, srv.URL, "deepseek-flash")

			before, err := json.Marshal(tc.messages)
			require.NoError(t, err)
			opts := append([]llms.CallOption{answerSchema()}, tc.opts...)
			for range 2 {
				_, err = llm.GenerateContent(t.Context(), tc.messages, opts...)
				require.NoError(t, err)
			}
			after, err := json.Marshal(tc.messages)
			require.NoError(t, err)
			assert.JSONEq(t, string(before), string(after), "the caller's messages must stay as they were")

			for call := range 2 {
				assertPlacement(t, srv.request(t, call), tc)
			}
		})
	}
}

// assertPlacement checks that exactly the expected wire message carries the
// schema instruction, once, after the text the caller wrote there.
func assertPlacement(t *testing.T, req wireRequest, tc placementCase) {
	t.Helper()

	roles := make([]string, len(req.Messages))
	for i, m := range req.Messages {
		roles[i] = m.Role
		if i != tc.instructed {
			assert.NotContains(t, string(m.Content), "JSON Schema", "message %d must not carry the schema", i)
		}
	}
	require.Equal(t, tc.roles, roles)

	got, array := messageText(t, req.Messages[tc.instructed].Content)
	assert.Equal(t, tc.array, array)
	assert.True(t, strings.HasPrefix(got, tc.prefix), "got %q", got)
	assert.Equal(t, 1, strings.Count(got, structuredoutput.SchemaInstruction), "a repeated call must not stack the instruction")
	assert.Contains(t, got, `"answer"`, "the schema itself travels")
	assert.Equal(t, tc.toolsNote, strings.Contains(got, structuredoutput.ToolsNote))
}

func TestTheFallbackReachesOnlyModelsWithoutJSONSchema(t *testing.T) {
	t.Parallel()

	human := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}

	t.Run("models with json_schema keep it", func(t *testing.T) {
		t.Parallel()
		for _, model := range []string{"gpt-4o", "glm-5-2", "mistral/zai-glm-5-2", "MiniMax-Text-01", "openrouter/minimax/minimax-m3"} {
			srv := newFallbackServer(t, fallbackReply{pieces: []string{`{"answer":"42"}`}})
			resp, err := newFallbackLLM(t, srv.URL, model).GenerateContent(t.Context(), human, answerSchema())
			require.NoError(t, err, model)

			req := srv.request(t, 0)
			assert.Contains(t, string(req.ResponseFormat), `"json_schema"`, model)
			text, _ := messageText(t, req.Messages[0].Content)
			assert.Equal(t, "question", text, model)
			for _, w := range resp.Warnings {
				assert.NotEqual(t, "WithStructuredOutput", w.Option, model)
			}
		}
	})

	t.Run("without the option the vendor is still refused before any request", func(t *testing.T) {
		t.Parallel()
		for _, model := range []string{"deepseek-flash", "MiniMax-M2.7"} {
			srv := newFallbackServer(t, fallbackReply{pieces: []string{`{"answer":"42"}`}})
			llm, err := New(WithBaseURL(srv.URL), WithToken("test"), WithModel(model))
			require.NoError(t, err)

			_, err = llm.GenerateContent(t.Context(), human, answerSchema())
			var unsupported *llms.ErrStructuredOutputUnsupported
			require.ErrorAs(t, err, &unsupported, model)
			assert.Zero(t, srv.requests(), model)
		}
	})

	t.Run("JSON mode alone is left as it was", func(t *testing.T) {
		t.Parallel()
		srv := newFallbackServer(t, fallbackReply{pieces: []string{`{"a":1}`}})
		resp, err := newFallbackLLM(t, srv.URL, "deepseek-flash").GenerateContent(t.Context(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "answer in json")}, llms.WithJSONMode())
		require.NoError(t, err)

		req := srv.request(t, 0)
		assert.JSONEq(t, `{"type":"json_object"}`, string(req.ResponseFormat))
		text, _ := messageText(t, req.Messages[0].Content)
		assert.Equal(t, "answer in json", text)
		assert.Empty(t, resp.Warnings)
	})
}

func TestTheFallbackChecksTheSchemaAsTheNativePathDoes(t *testing.T) {
	t.Parallel()

	human := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}
	for _, tc := range []struct {
		name   string
		config llms.StructuredOutputConfig
		opts   []Option
		want   error
	}{
		{name: "a schema without a name", config: llms.StructuredOutputConfig{Schema: objectSchema()}, want: llms.ErrStructuredOutputConfig},
		{
			name:   "a schema outside the strict subset",
			config: llms.StructuredOutputConfig{Name: "answer", Schema: json.RawMessage(`{"type":"array","items":{"type":"string"}}`)},
			want:   llms.ErrStructuredOutputConfig,
		},
		{
			name:   "a client-level response format",
			config: llms.StructuredOutputConfig{Name: "answer", Schema: objectSchema()},
			opts:   []Option{WithResponseFormat(ResponseFormatJSON)},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var errs [2]error
			for i, model := range []string{"gpt-4o", "deepseek-flash"} {
				srv := newFallbackServer(t, fallbackReply{pieces: []string{`{"answer":"42"}`}})
				_, errs[i] = newFallbackLLM(t, srv.URL, model, tc.opts...).GenerateContent(t.Context(), human,
					llms.WithStructuredOutput(tc.config))
				require.Error(t, errs[i], model)
				assert.Zero(t, srv.requests(), "%s: refused before any request", model)
			}
			if tc.want != nil {
				require.ErrorIs(t, errs[1], tc.want)
			} else {
				var conflict *llms.ErrStructuredOutputConflict
				require.ErrorAs(t, errs[1], &conflict)
			}
			assert.Equal(t, errs[0].Error(), errs[1].Error(), "the vendor without json_schema gets the same verdict")
		})
	}
}

func TestTheFallbackValidatesTheAnswerLocally(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name    string
		reply   fallbackReply
		invalid bool
		want    string
	}{
		{name: "a valid answer", reply: fallbackReply{pieces: []string{`{"answer":"42"}`}}, want: `{"answer":"42"}`},
		{name: "a fenced answer is unwrapped", reply: fallbackReply{pieces: []string{"```json\n{\"answer\":\"42\"}\n```"}}, want: `{"answer":"42"}`},
		{name: "an answer off the schema", reply: fallbackReply{pieces: []string{`{"answer":42}`}}, invalid: true, want: `{"answer":42}`},
		{name: "prose around the answer", reply: fallbackReply{pieces: []string{`Here: {"answer":"42"}`}}, invalid: true, want: `Here: {"answer":"42"}`},
		{name: "broken JSON", reply: fallbackReply{pieces: []string{`{"answer":["42"}`}}, invalid: true, want: `{"answer":["42"}`},
		{name: "an empty answer", reply: fallbackReply{pieces: []string{""}}, invalid: true},
		{
			name:  "a truncated answer is left alone",
			reply: fallbackReply{pieces: []string{"```json\n{\"answer\":\"```"}, finishReason: "length"}, want: "```json\n{\"answer\":\"```",
		},
		{name: "a tool-call turn is not the final answer", reply: fallbackReply{pieces: []string{"   "}, finishReason: "tool_calls", toolCall: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			srv := newFallbackServer(t, tc.reply)
			opts := []llms.CallOption{answerSchema()}
			if tc.reply.toolCall {
				opts = append(opts, llms.WithTools([]llms.Tool{weatherTool()}))
			}
			resp, err := newFallbackLLM(t, srv.URL, "glm-5.3").GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}, opts...)
			require.NotNil(t, resp, "the answer comes back with any error")

			if tc.invalid {
				var validation *llms.ErrStructuredOutputValidation
				require.ErrorAs(t, err, &validation)
				assert.Equal(t, "glm-5.3", validation.Model)
			} else {
				require.NoError(t, err)
			}
			if !tc.reply.toolCall {
				assert.Equal(t, tc.want, resp.Choices[0].Content)
			}
		})
	}
}

func TestTheFallbackValidatesTheStreamedAnswer(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name    string
		pieces  []string
		content string
		invalid bool
	}{
		{name: "a valid answer in pieces", pieces: []string{`{"answer":`, `"42"}`}, content: `{"answer":"42"}`},
		{name: "a fenced answer in pieces is unwrapped once complete", pieces: []string{"```json\n", `{"answer":"42"}`, "\n```"}, content: `{"answer":"42"}`},
		{name: "prose in pieces", pieces: []string{"The answer ", "is 42."}, content: "The answer is 42.", invalid: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			srv := newFallbackServer(t, fallbackReply{pieces: tc.pieces})
			var streamed strings.Builder
			resp, err := newFallbackLLM(t, srv.URL, "deepseek-flash").GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")},
				answerSchema(), llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
					if chunk.Type == streaming.ChunkTypeText {
						streamed.WriteString(chunk.Content)
					}
					return nil
				}))

			require.NotNil(t, resp)
			assert.Equal(t, strings.Join(tc.pieces, ""), streamed.String(), "every chunk reaches the caller as sent")
			assert.Equal(t, tc.content, resp.Choices[0].Content)
			var validation *llms.ErrStructuredOutputValidation
			assert.Equal(t, tc.invalid, errors.As(err, &validation), "got %v", err)
			if !tc.invalid {
				require.NoError(t, err)
			}
		})
	}
}

func TestTheFallbackValidatesMiniMaxStreamedAnswerAfterItsThinking(t *testing.T) {
	t.Parallel()

	pieces := []string{"<think>\nThe user", " wants JSON.", "\n</think>\n\n", "```json\n", `{"answer":"42"}`, "\n```"}
	for _, tc := range []struct {
		name       string
		reasonings []string
	}{
		{name: "thinking only inside the content"},
		// What MiniMax's API streams: every thinking piece twice, inside the
		// content and as a reasoning delta, so the client leaves the content whole.
		{name: "thinking in the content and as reasoning too", reasonings: []string{"The user", " wants JSON."}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			srv := newFallbackServer(t, fallbackReply{pieces: pieces, reasonings: tc.reasonings})
			var thought strings.Builder
			resp, err := newFallbackLLM(t, srv.URL, "MiniMax-M2.7").GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")},
				answerSchema(), llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
					if chunk.Type == streaming.ChunkTypeReasoning {
						thought.WriteString(chunk.Reasoning.Content)
					}
					return nil
				}))

			require.NoError(t, err)
			assert.Equal(t, `{"answer":"42"}`, resp.Choices[0].Content)
			require.NotNil(t, resp.Choices[0].Reasoning)
			assert.Contains(t, resp.Choices[0].Reasoning.Content, "The user wants JSON.")
			assert.NotContains(t, resp.Choices[0].Reasoning.Content, "think>")
			assert.Contains(t, thought.String(), "wants JSON")
			assert.Empty(t, srv.request(t, 0).ResponseFormat)
		})
	}
}

func TestTheFallbackMovesOnlyTheThinkingAtTheHeadOfTheAnswer(t *testing.T) {
	t.Parallel()

	// Every piece carries a reasoning delta, so the client leaves the content whole.
	for name, pieces := range map[string][]string{
		"thinking after the answer": {`{"answer":"42"} `, "<think>late</think>"},
		"thinking never closed":     {"<think>never closed ", `{"answer":"42"}`},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			srv := newFallbackServer(t, fallbackReply{pieces: pieces, reasonings: []string{"r", "r"}})
			resp, err := newFallbackLLM(t, srv.URL, "MiniMax-M2.7").GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}, answerSchema(),
				llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

			var validation *llms.ErrStructuredOutputValidation
			require.ErrorAs(t, err, &validation)
			require.NotNil(t, resp)
			assert.Equal(t, strings.Join(pieces, ""), resp.Choices[0].Content)
		})
	}
}

// A thinking block at the head of the answer is taken out in every shape a
// stream can leave it in, and never replaces the reasoning that was streamed.
func TestTheFallbackTakesOutTheThinkingAtTheHeadOfTheAnswer(t *testing.T) {
	t.Parallel()

	const answer = `{"answer":"42"}`
	for _, tc := range []struct {
		name       string
		pieces     []string
		reasonings []string
		want       string
	}{
		{name: "a thinking block", pieces: []string{"<thinking>plan</thinking>\n\n", answer}, reasonings: []string{"plan"}, want: "plan"},
		{name: "whitespace before the block", pieces: []string{"\n<think>plan</think>\n\n", answer}, reasonings: []string{"plan"}, want: "plan"},
		// The chunk splitter sees no tag in either piece, and no reasoning delta
		// came, so the block reaches the unwrap whole with no reasoning beside it.
		{name: "a tag split across chunks without reasoning deltas", pieces: []string{"<thi", "nk>plan</think>\n\n", answer}, want: "plan"},
		{name: "streamed reasoning differing from the block", pieces: []string{"<think>plan</think>\n\n", answer}, reasonings: []string{"summary"}, want: "summary"},
		// As MiniMax-M2.5 streams: no reasoning deltas, so the chunk splitter
		// takes the block and leaves the blank lines after it in the answer.
		{name: "a block the client already split", pieces: []string{"<think>plan", "</think>\n\n", answer}, want: "plan"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			srv := newFallbackServer(t, fallbackReply{pieces: tc.pieces, reasonings: tc.reasonings})
			resp, err := newFallbackLLM(t, srv.URL, "MiniMax-M2.7").GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}, answerSchema(),
				llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))
			require.NoError(t, err)
			assert.Equal(t, answer, resp.Choices[0].Content)
			require.NotNil(t, resp.Choices[0].Reasoning)
			assert.Equal(t, tc.want, strings.TrimSpace(resp.Choices[0].Reasoning.Content))
		})
	}
}

// Only an emulated structured-output answer is unwrapped: every other answer,
// a native json_schema one included, comes back and is validated as sent.
func TestAnswersOutsideTheFallbackStayAsSent(t *testing.T) {
	t.Parallel()

	const fenced = "```json\n{\"answer\":\"42\"}\n```"
	human := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}

	t.Run("JSON mode alone on a fallback client", func(t *testing.T) {
		t.Parallel()
		srv := newFallbackServer(t, fallbackReply{pieces: []string{fenced}})
		resp, err := newFallbackLLM(t, srv.URL, "deepseek-flash").GenerateContent(t.Context(), human, llms.WithJSONMode())
		require.NoError(t, err)
		assert.Equal(t, fenced, resp.Choices[0].Content)
	})

	t.Run("a native json_schema answer is validated as sent", func(t *testing.T) {
		t.Parallel()
		srv := newFallbackServer(t, fallbackReply{pieces: []string{fenced}})
		resp, err := newFallbackLLM(t, srv.URL, "gpt-4o").GenerateContent(t.Context(), human, answerSchema())
		var validation *llms.ErrStructuredOutputValidation
		require.ErrorAs(t, err, &validation)
		assert.Equal(t, fenced, resp.Choices[0].Content)
	})
}

// The per-call model decides the unwrap, as it decides the request.
func TestThePerCallModelDecidesTheFallbackUnwrap(t *testing.T) {
	t.Parallel()

	srv := newFallbackServer(t, fallbackReply{pieces: []string{"```json\n{\"answer\":\"42\"}\n```"}})
	resp, err := newFallbackLLM(t, srv.URL, "gpt-4o").GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")},
		answerSchema(), llms.WithModel("minimax/MiniMax-M2.7"))
	require.NoError(t, err)
	assert.Equal(t, `{"answer":"42"}`, resp.Choices[0].Content)
	assert.Empty(t, srv.request(t, 0).ResponseFormat)
}
