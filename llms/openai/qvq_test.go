package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

type streamDoer struct{ bodies [][]byte }

func (d *streamDoer) Do(req *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(req.Body)
	d.bodies = append(d.bodies, body)
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body: io.NopCloser(strings.NewReader(`data: {"id":"x","object":"chat.completion.chunk","created":1,` +
			`"model":"qvq-max","choices":[{"index":0,"delta":{"role":"assistant","content":"a cat"},` +
			`"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n")),
	}, nil
}

func TestQVQIsAskedOnlyForWhatDashScopeServesIt(t *testing.T) {
	t.Parallel()

	const dashScope = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	stream := llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil })
	call := func(opts ...llms.CallOption) (*streamDoer, error) {
		doer := &streamDoer{}
		llm := newUnitLLM(t, WithBaseURL(dashScope), WithModel("qvq-max"), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what is in the picture?")}, opts...)
		return doer, err
	}

	doer, err := call()
	var streamOnly *reasoning.ErrThinkingRequiresStream
	require.True(t, errors.As(err, &streamOnly), "QVQ streams only: %v", err)
	require.Empty(t, doer.bodies, "refused before the network")

	doer, err = call(stream, llms.WithReasoningDisabled())
	var off *reasoning.ErrReasoningOffUnsupported
	require.True(t, errors.As(err, &off), "QVQ only thinks: %v", err)
	require.Empty(t, doer.bodies)

	doer, err = call(stream, llms.WithReasoning(llms.ReasoningHigh, 0))
	require.NoError(t, err)
	require.Len(t, doer.bodies, 1)
	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.bodies[0], &body))
	assert.NotContains(t, body, "reasoning_effort", "DashScope documents no effort for QVQ")
	assert.Equal(t, true, body["stream"])
}

func TestOnlyModelStudioHoldsQVQToAStream(t *testing.T) {
	t.Parallel()

	call := func(baseURL, model string) (*bodyDoer, error) {
		doer := &bodyDoer{}
		llm := newUnitLLM(t, WithBaseURL(baseURL), WithModel(model), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what is in the picture?")})
		return doer, err
	}

	doer, err := call("http://litellm.internal/v1", "dashscope/qvq-max")
	var streamOnly *reasoning.ErrThinkingRequiresStream
	require.True(t, errors.As(err, &streamOnly), "the gateway's dashscope route is Model Studio: %v", err)
	require.Nil(t, doer.body)

	doer, err = call("http://vllm.internal:8000/v1", "qvq-72b-preview")
	require.NoError(t, err, "open weights on another host answer without a stream")
	var body map[string]any
	require.NoError(t, json.Unmarshal(doer.body, &body))
	assert.Equal(t, "qvq-72b-preview", body["model"])
	assert.NotContains(t, body, "stream")
}

func TestQVQAndQwQKeepTheirThinkingInEverySpelling(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ baseURL, model string }{
		{"http://localhost:11434/v1", "qvq:72b"},
		{"http://localhost:11434/v1", "qvq"},
		{"http://localhost:11434/v1", "qwq:32b"},
		{"http://localhost:11434/v1", "qwq"},
		{"http://vllm.internal:8000/v1", "Qwen/QVQ-72B-Preview"},
		{"http://vllm.internal:8000/v1", "qvq-72b-preview"},
		{"http://vllm.internal:8000/v1", "qwq-32b"},
		{"https://api.groq.com/openai/v1", "qwen-qwq-32b"},
		{"http://localhost:1234/v1", "qwen_qwq-32b"},
	} {
		doer := &bodyDoer{}
		llm := newUnitLLM(t, WithBaseURL(tc.baseURL), WithModel(tc.model), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, llms.WithReasoningDisabled())
		var off *reasoning.ErrReasoningOffUnsupported
		require.True(t, errors.As(err, &off), "%s on %s only thinks: %v", tc.model, tc.baseURL, err)
		require.Nil(t, doer.body, "%s: refused before the network", tc.model)
	}

	body, warnings := hostCall(t, "http://vllm.internal:8000/v1", "qvq-72b-preview", llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.NotContains(t, body, "reasoning_effort", "no host documents an effort for QVQ")
	assert.Equal(t, llms.WarningDrop, warnings["WithReasoning"].Kind)
}

func TestQVQOnModelStudioIsRefusedAForcedToolChoice(t *testing.T) {
	t.Parallel()

	stream := llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil })
	const dashScope = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	for _, tc := range []struct {
		baseURL, model string
		choice         any
	}{
		{dashScope, "qvq-max", "required"},
		{dashScope, "qvq-plus", "required"},
		{dashScope, "qvq-max", map[string]any{"type": "function", "name": "lookup"}},
		{"http://litellm.internal/v1", "dashscope/qvq-max", "required"},
	} {
		body, err := callWithATool(t, tc.baseURL, tc.model, stream, llms.WithToolChoice(tc.choice))
		var refused *reasoning.ErrForcedToolChoiceUnsupported
		require.True(t, errors.As(err, &refused), "%s %v: %v", tc.model, tc.choice, err)
		require.Nil(t, body, "%s: refused before the network", tc.model)
	}

	for _, model := range []string{"qvq-max", "qwq-plus"} {
		body, err := callWithATool(t, dashScope, model, stream,
			llms.WithToolChoice(map[string]any{"type": "function", "name": "lookup"}),
			llms.WithExtraBody(map[string]any{"enable_thinking": false}))
		var refused *reasoning.ErrForcedToolChoiceUnsupported
		require.True(t, errors.As(err, &refused), "%s keeps thinking whatever the extra body says: %v", model, err)
		require.Nil(t, body, model)
	}

	for _, tc := range []struct {
		baseURL, model string
		choice         any
	}{
		{dashScope, "qvq-max", "auto"},
		{"http://vllm.internal:8000/v1", "qvq-72b-preview", "required"},
		{"https://ai-gateway.vercel.sh/v1", "dashscope/qvq-max", "required"},
	} {
		doer := &streamDoer{}
		llm := newUnitLLM(t, WithBaseURL(tc.baseURL), WithModel(tc.model), WithHTTPClient(doer))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "what is in the picture?")},
			stream, llms.WithTools([]llms.Tool{astraTool()}), llms.WithToolChoice(tc.choice))
		require.NoError(t, err, tc.model)
		require.Len(t, doer.bodies, 1, tc.model)
		var body map[string]any
		require.NoError(t, json.Unmarshal(doer.bodies[0], &body))
		assert.Equal(t, tc.choice, body["tool_choice"], "%s on %s", tc.model, tc.baseURL)
	}
}
