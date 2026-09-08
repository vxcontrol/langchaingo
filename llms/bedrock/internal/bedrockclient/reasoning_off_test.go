package bedrockclient

import (
	"context"
	"net/http"
	"sync/atomic"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type countingTransport struct{ seen *int32 }

func (c *countingTransport) Do(req *http.Request) (*http.Response, error) {
	atomic.AddInt32(c.seen, 1)
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       http.NoBody,
		Header:     http.Header{},
		Request:    req,
	}, nil
}

func legacyClientCounting(seen *int32) *Client {
	return &Client{client: bedrockruntime.New(bedrockruntime.Options{
		Region:           "us-east-1",
		Credentials:      credentials.NewStaticCredentialsProvider("k", "s", ""),
		HTTPClient:       &countingTransport{seen: seen},
		RetryMaxAttempts: 1,
	})}
}

func offCall(t *testing.T, model string) (int32, error) {
	t.Helper()

	var seen int32
	_, err := legacyClientCounting(&seen).CreateCompletion(context.Background(), model,
		[]Message{{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"}},
		llms.CallOptions{Model: aws.String(model), Reasoning: &llms.ReasoningConfig{Mode: llms.ReasoningOff}},
	)
	return atomic.LoadInt32(&seen), err
}

func TestTheLegacyDoorRefusesAnUndisableableThinking(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"us.deepseek.r1-v1:0",
		"us.anthropic.claude-sonnet-5-v1:0",
	} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			seen, err := offCall(t, model)

			var offErr *reasoning.ErrReasoningOffUnsupported
			require.ErrorAs(t, err, &offErr,
				"the Converse door refuses this model; the default door must refuse it too")
			assert.Zero(t, seen, "the refusal must come before the request leaves")
		})
	}
}

func TestTheLegacyDoorServesAnOffThatOmittingAlreadyMeans(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"meta.llama3-70b-instruct-v1:0",
		"us.amazon.nova-2-lite-v1:0",
	} {
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			_, err := offCall(t, model)

			var offErr *reasoning.ErrReasoningOffUnsupported
			if err != nil {
				require.NotErrorAs(t, err, &offErr,
					"omitting the field already means off here, so the caller loses nothing")
			}
		})
	}
}

func TestTheLegacyDoorLeavesACallThatSaidNothingAboutThinkingAlone(t *testing.T) {
	t.Parallel()

	model := "us.anthropic.claude-sonnet-5-v1:0"
	var seen int32
	_, err := legacyClientCounting(&seen).CreateCompletion(context.Background(), model,
		[]Message{{Role: llms.ChatMessageTypeHuman, Content: "hi", Type: "text"}},
		llms.CallOptions{Model: aws.String(model)},
	)

	var offErr *reasoning.ErrReasoningOffUnsupported
	require.NotErrorAs(t, err, &offErr,
		"a caller who said nothing about thinking asked for nothing that could be refused")
	assert.Positive(t, atomic.LoadInt32(&seen), "the call must reach the vendor")
}
