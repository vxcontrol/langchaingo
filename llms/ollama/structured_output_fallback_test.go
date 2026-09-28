package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/callbacks"
	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
	"github.com/vxcontrol/langchaingo/llms/structuredoutput"
)

// fallbackServer answers every chat request with the same NDJSON lines and
// keeps the request bodies it received for the test goroutine to decode.
type fallbackServer struct {
	*httptest.Server

	mu     sync.Mutex
	bodies [][]byte
}

func newFallbackServer(t *testing.T, lines ...string) *fallbackServer {
	t.Helper()

	fs := &fallbackServer{}
	fs.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		fs.mu.Lock()
		fs.bodies = append(fs.bodies, body)
		fs.mu.Unlock()

		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, line := range lines {
			_, _ = io.WriteString(w, line+"\n")
		}
	}))
	t.Cleanup(fs.Close)
	return fs
}

// wireRequest is the part of a chat request these tests read.
type wireRequest struct {
	Format   json.RawMessage `json:"format"`
	Messages []wireMessage   `json:"messages"`
	Tools    []any           `json:"tools"`
}

type wireMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images"`
}

func (fs *fallbackServer) requests() int {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	return len(fs.bodies)
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

// chatLine is one NDJSON line of an Ollama chat response.
func chatLine(content string, done bool, doneReason string) string {
	line, _ := json.Marshal(map[string]any{
		"model":       "m",
		"message":     map[string]any{"role": "assistant", "content": content},
		"done":        done,
		"done_reason": doneReason,
	})
	return string(line)
}

func answerLine(content string) string { return chatLine(content, true, "stop") }

func newFallbackLLM(t *testing.T, serverURL, model string, extra ...Option) *LLM {
	t.Helper()

	opts := append([]Option{WithServerURL(serverURL), WithModel(model), WithCloudStructuredOutputFallback()}, extra...)
	llm, err := New(opts...)
	require.NoError(t, err)
	return llm
}

func answerSchema() llms.CallOption {
	return llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "answer", Schema: json.RawMessage(ollamaSOSchema)})
}

// instructed reports whether a wire message carries the injected instruction.
func instructed(content string) bool {
	return strings.Contains(content, structuredoutput.SchemaInstruction) && strings.Contains(content, ollamaSOSchema)
}

// placementCase is a conversation and where the fallback must put the schema.
type placementCase struct {
	name       string
	messages   []llms.MessageContent
	opts       []llms.CallOption
	roles      []string
	instructed int
	prefix     string
	toolsNote  bool
}

func placementCases() []placementCase {
	weatherTool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name:       "get_weather",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
	}}
	toolCall := llms.ToolCall{
		ID: makeToolCallID(0, "get_weather"), Type: "function",
		FunctionCall: &llms.FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`},
	}

	return []placementCase{
		{
			name:     "the only user turn",
			messages: []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")},
			roles:    []string{"user"}, instructed: 0, prefix: "question\n\n",
		},
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
			opts:  []llms.CallOption{llms.WithTools([]llms.Tool{weatherTool})},
			roles: []string{"user", "assistant", "tool"}, instructed: 0, prefix: "weather in Paris?\n\n", toolsNote: true,
		},
		{
			name:     "a user turn of its own when there is none, since the cloud answers no conversation without one",
			messages: []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeSystem, "sys")},
			roles:    []string{"system", "user"}, instructed: 1, prefix: structuredoutput.SchemaInstruction,
		},
		{
			name: "an image-only user turn without a blank separator",
			messages: []llms.MessageContent{{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{
				llms.BinaryContent{MIMEType: "image/png", Data: []byte("png")},
			}}},
			roles: []string{"user"}, instructed: 0, prefix: structuredoutput.SchemaInstruction,
		},
	}
}

func TestTheCloudFallbackCarriesTheSchemaInThePrompt(t *testing.T) {
	t.Parallel()

	for _, tc := range placementCases() {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			srv := newFallbackServer(t, answerLine(`{"answer":"42"}`))
			llm := newFallbackLLM(t, srv.URL, "gpt-oss:120b-cloud")

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

	assert.JSONEq(t, `""`, string(req.Format), "the cloud gets the empty format")

	roles := make([]string, len(req.Messages))
	for i, m := range req.Messages {
		roles[i] = m.Role
		if i != tc.instructed {
			assert.False(t, instructed(m.Content), "message %d must not carry the schema", i)
		}
	}
	require.Equal(t, tc.roles, roles)

	got := req.Messages[tc.instructed].Content
	assert.True(t, instructed(got), "message %d must carry the schema: %q", tc.instructed, got)
	assert.Equal(t, 1, strings.Count(got, ollamaSOSchema), "a repeated call must not stack the instruction")
	assert.True(t, strings.HasPrefix(got, tc.prefix), "got %q", got)
	assert.Equal(t, tc.toolsNote, strings.Contains(got, structuredoutput.ToolsNote))
	// The phrases that make a model drop a caller's Markdown or fences, measured
	// against Ollama Cloud; a weaker wording loses to a conflicting system prompt.
	assert.Contains(t, got, "overrides any earlier instruction about format or style")
	assert.Contains(t, got, "no code fences")
}

// redirectTransport sends every request to target, whatever host it named.
type redirectTransport struct{ target *url.URL }

func (r redirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.URL.Scheme, req.URL.Host = r.target.Scheme, r.target.Host
	return http.DefaultTransport.RoundTrip(req)
}

func TestTheCloudFallbackReachesOnlyCloudServedModels(t *testing.T) {
	t.Parallel()

	human := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}

	t.Run("a local model keeps the native schema", func(t *testing.T) {
		t.Parallel()

		srv := newFallbackServer(t, answerLine(`{"answer":"42"}`))
		resp, err := newFallbackLLM(t, srv.URL, "gpt-oss:120b").GenerateContent(t.Context(), human, answerSchema())
		require.NoError(t, err)

		req := srv.request(t, 0)
		assert.JSONEq(t, ollamaSOSchema, string(req.Format))
		assert.Equal(t, "question", req.Messages[0].Content)
		assert.Empty(t, resp.Warnings)
	})

	t.Run("a per-call cloud model is emulated, and a per-call local model is not", func(t *testing.T) {
		t.Parallel()

		srv := newFallbackServer(t, answerLine(`{"answer":"42"}`))
		resp, err := newFallbackLLM(t, srv.URL, "gpt-oss:120b").GenerateContent(t.Context(), human,
			answerSchema(), llms.WithModel("gpt-oss:120b-cloud"))
		require.NoError(t, err)
		assert.Equal(t, "gpt-oss:120b-cloud", ollamaWarningsByOption(resp.Warnings)["WithStructuredOutput"].Model,
			"the warning names the model the call ran on")
		_, err = newFallbackLLM(t, srv.URL, "glm-4.6:cloud").GenerateContent(t.Context(), human,
			answerSchema(), llms.WithModel("llama3"))
		require.NoError(t, err)

		cloud, local := srv.request(t, 0), srv.request(t, 1)
		assert.JSONEq(t, `""`, string(cloud.Format))
		assert.True(t, instructed(cloud.Messages[0].Content))
		assert.JSONEq(t, ollamaSOSchema, string(local.Format))
		assert.Equal(t, "question", local.Messages[0].Content)
	})

	t.Run("a model on an ollama.com host is emulated whatever its tag", func(t *testing.T) {
		t.Parallel()

		srv := newFallbackServer(t, answerLine(`{"answer":"42"}`))
		target, err := url.Parse(srv.URL)
		require.NoError(t, err)
		llm := newFallbackLLM(t, "https://api.ollama.com", "gpt-oss:120b",
			WithHTTPClient(&http.Client{Transport: redirectTransport{target: target}}))

		resp, err := llm.GenerateContent(t.Context(), human, answerSchema())
		require.NoError(t, err)
		assert.JSONEq(t, `""`, string(srv.request(t, 0).Format))
		assert.True(t, instructed(srv.request(t, 0).Messages[0].Content))
		assert.Equal(t, llms.WarningSubstitute, ollamaWarningsByOption(resp.Warnings)["WithStructuredOutput"].Kind)
	})

	t.Run("JSON mode alone is still dropped and gets no instruction", func(t *testing.T) {
		t.Parallel()

		srv := newFallbackServer(t, answerLine(`{}`))
		resp, err := newFallbackLLM(t, srv.URL, "gpt-oss:120b-cloud").GenerateContent(t.Context(), human, llms.WithJSONMode())
		require.NoError(t, err)

		req := srv.request(t, 0)
		assert.JSONEq(t, `""`, string(req.Format))
		assert.Equal(t, "question", req.Messages[0].Content)
		assert.Equal(t, llms.WarningDrop, ollamaWarningsByOption(resp.Warnings)["WithJSONMode"].Kind)
		assert.NotContains(t, ollamaWarningsByOption(resp.Warnings), "WithStructuredOutput")
	})

	t.Run("without the option the cloud still refuses the schema before any request", func(t *testing.T) {
		t.Parallel()

		srv := newFallbackServer(t, answerLine(`{"answer":"42"}`))
		llm, err := New(WithServerURL(srv.URL), WithModel("gpt-oss:120b-cloud"))
		require.NoError(t, err)

		_, err = llm.GenerateContent(t.Context(), human, answerSchema())
		var unsupported *llms.ErrStructuredOutputUnsupported
		require.ErrorAs(t, err, &unsupported)
		assert.Zero(t, srv.requests())
	})

	t.Run("an invalid schema config is refused before any request", func(t *testing.T) {
		t.Parallel()

		srv := newFallbackServer(t, answerLine(`{"answer":"42"}`))
		_, err := newFallbackLLM(t, srv.URL, "gpt-oss:120b-cloud").GenerateContent(t.Context(), human,
			llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "answer", Schema: json.RawMessage(`{"type":`)}))
		require.ErrorIs(t, err, llms.ErrStructuredOutputConfig)
		assert.Zero(t, srv.requests())
	})
}

func TestATruncatedFencedAnswerIsNotUnwrapped(t *testing.T) {
	t.Parallel()

	raw := "```json\n{\"answer\":\"```"
	srv := newFallbackServer(t, chatLine(raw, true, "length"))
	resp, err := newFallbackLLM(t, srv.URL, "gpt-oss:120b-cloud").GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}, answerSchema())
	require.NoError(t, err)
	assert.Equal(t, raw, resp.Choices[0].Content, "an answer that is not judged keeps its raw text")
}

func TestALocalModelsFencedAnswerIsNotUnwrapped(t *testing.T) {
	t.Parallel()

	srv := newFallbackServer(t, answerLine("```json\n{\"answer\":\"42\"}\n```"))
	resp, err := newFallbackLLM(t, srv.URL, "gpt-oss:120b").GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}, answerSchema())

	var validation *llms.ErrStructuredOutputValidation
	require.ErrorAs(t, err, &validation, "the native schema constrains the server; nothing is repaired locally")
	assert.Equal(t, "```json\n{\"answer\":\"42\"}\n```", resp.Choices[0].Content)
}

func TestTheCloudFallbackValidatesTheAnswerLocally(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name       string
		content    string
		doneReason string
		opts       []llms.CallOption
		invalid    bool
		truncated  bool
		want       string
	}{
		{name: "a valid answer", content: `{"answer":"42"}`, want: `{"answer":"42"}`},
		{name: "surrounding whitespace", content: "\n {\"answer\":\"42\"} \n", want: `{"answer":"42"}`},
		{name: "thinking tags before the answer", content: `<think>plan</think>{"answer":"42"}`, want: `{"answer":"42"}`},
		{name: "an answer off the schema", content: `{"answer":42}`, invalid: true},
		{name: "a fenced answer is unwrapped", content: "```json\n{\"answer\":\"42\"}\n```", want: `{"answer":"42"}`},
		{name: "an untagged fence is unwrapped", content: "\n```\n{\"answer\":\"42\"}\n```\n", want: `{"answer":"42"}`},
		{
			name: "an unwrapped answer off the schema", content: "```json\n{\"answer\":42}\n```",
			invalid: true, want: `{"answer":42}`,
		},
		{name: "prose around a fence", content: "Here it is:\n```json\n{\"answer\":\"42\"}\n```", invalid: true},
		{name: "two fences", content: "```json\n{\"answer\":\"a\"}\n```\n```json\n{\"answer\":\"b\"}\n```", invalid: true},
		{name: "a fence of another language", content: "```yaml\nanswer: \"42\"\n```", invalid: true},
		{name: "prose around the answer", content: `Here it is: {"answer":"42"}`, invalid: true},
		{name: "two values", content: `{"answer":"a"}{"answer":"b"}`, invalid: true},
		{name: "an empty answer", content: "", invalid: true},
		{name: "a truncated answer is left alone", content: `{"answer":"4`, doneReason: "length", truncated: true, want: `{"answer":"4`},
		{
			name: "a truncated answer fails when the caller asks", content: `{"answer":"4`, doneReason: "length",
			opts: []llms.CallOption{llms.WithFailOnTruncation()}, truncated: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			doneReason := tc.doneReason
			if doneReason == "" {
				doneReason = "stop"
			}
			srv := newFallbackServer(t, chatLine(tc.content, true, doneReason))
			llm := newFallbackLLM(t, srv.URL, "gpt-oss:120b-cloud")

			opts := append([]llms.CallOption{answerSchema()}, tc.opts...)
			resp, err := llm.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}, opts...)

			require.NotNil(t, resp, "the answer comes back with any error")
			warning := ollamaWarningsByOption(resp.Warnings)["WithStructuredOutput"]
			assert.Equal(t, llms.WarningSubstitute, warning.Kind)
			assert.Equal(t, "answer", warning.Asked)
			assert.Equal(t, "a prompt instruction", warning.Sent)
			assert.NotEmpty(t, warning.Reason)

			switch {
			case tc.invalid:
				var validation *llms.ErrStructuredOutputValidation
				require.ErrorAs(t, err, &validation)
				assert.Equal(t, "gpt-oss:120b-cloud", validation.Model)
				want := tc.content
				if tc.want != "" {
					want = tc.want
				}
				assert.Equal(t, want, resp.Choices[0].Content, "only a single whole fence is removed, nothing else is repaired")
			case len(tc.opts) > 0:
				assert.True(t, llms.IsTruncatedError(err), "got %v", err)
			default:
				require.NoError(t, err)
				assert.Equal(t, tc.want, resp.Choices[0].Content)
			}
			assert.Equal(t, tc.truncated, resp.Choices[0].Truncated)
		})
	}
}

func TestTheCloudFallbackLetsAToolCallThrough(t *testing.T) {
	t.Parallel()

	srv := newFallbackServer(t, `{"model":"m","message":{"role":"assistant","content":"",`+
		`"tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Paris"}}}]},"done":true,"done_reason":"stop"}`)
	llm := newFallbackLLM(t, srv.URL, "gpt-oss:120b-cloud")

	resp, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "weather in Paris?")},
		answerSchema(), llms.WithTools([]llms.Tool{{Type: "function", Function: &llms.FunctionDefinition{
			Name:       "get_weather",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
		}}}))
	require.NoError(t, err, "a tool-call turn is not the final answer and is not validated")
	require.Len(t, resp.Choices[0].ToolCalls, 1)
	assert.Equal(t, "get_weather", resp.Choices[0].ToolCalls[0].FunctionCall.Name)

	req := srv.request(t, 0)
	assert.Len(t, req.Tools, 1)
	assert.Contains(t, req.Messages[0].Content, structuredoutput.ToolsNote)
}

func TestTheCloudFallbackValidatesTheStreamedAnswer(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name    string
		chunks  []string
		content string
		invalid bool
	}{
		{name: "a valid answer in pieces", chunks: []string{`{"answer":`, `"42"}`}, content: `{"answer":"42"}`},
		{
			name:    "a fenced answer in pieces is unwrapped once complete",
			chunks:  []string{"```json\n", `{"answer":"42"}`, "\n```"},
			content: `{"answer":"42"}`,
		},
		{name: "prose in pieces", chunks: []string{"The answer ", "is 42."}, content: "The answer is 42.", invalid: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			lines := make([]string, 0, len(tc.chunks)+1)
			for _, chunk := range tc.chunks {
				lines = append(lines, chatLine(chunk, false, ""))
			}
			srv := newFallbackServer(t, append(lines, chatLine("", true, "stop"))...)
			llm := newFallbackLLM(t, srv.URL, "gpt-oss:120b-cloud")

			var streamed strings.Builder
			resp, err := llm.GenerateContent(t.Context(),
				[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")},
				answerSchema(), llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
					if chunk.Type == streaming.ChunkTypeText {
						streamed.WriteString(chunk.Content)
					}
					return nil
				}))

			require.NotNil(t, resp)
			assert.Equal(t, strings.Join(tc.chunks, ""), streamed.String(), "every chunk reaches the caller as sent")
			assert.Equal(t, tc.content, resp.Choices[0].Content)
			assert.JSONEq(t, `""`, string(srv.request(t, 0).Format))
			assert.Equal(t, llms.WarningSubstitute, ollamaWarningsByOption(resp.Warnings)["WithStructuredOutput"].Kind)

			var validation *llms.ErrStructuredOutputValidation
			assert.Equal(t, tc.invalid, errors.As(err, &validation), "got %v", err)
			if !tc.invalid {
				require.NoError(t, err)
			}
		})
	}
}

// countingHandler counts the closing callbacks of each call.
type countingHandler struct {
	callbacks.SimpleHandler

	errs []error
	ends int
}

func (h *countingHandler) HandleLLMError(_ context.Context, err error) { h.errs = append(h.errs, err) }

func (h *countingHandler) HandleLLMGenerateContentEnd(context.Context, *llms.ContentResponse) {
	h.ends++
}

func TestAFailedCloudFallbackValidationClosesTheCallWithAnError(t *testing.T) {
	t.Parallel()

	srv := newFallbackServer(t, answerLine("not JSON"))
	llm := newFallbackLLM(t, srv.URL, "gpt-oss:120b-cloud")
	handler := &countingHandler{}
	llm.CallbacksHandler = handler

	_, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}, answerSchema())

	var validation *llms.ErrStructuredOutputValidation
	require.ErrorAs(t, err, &validation)
	require.Len(t, handler.errs, 1)
	assert.ErrorAs(t, handler.errs[0], &validation)
	assert.Zero(t, handler.ends)
}
