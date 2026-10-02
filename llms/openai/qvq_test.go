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
