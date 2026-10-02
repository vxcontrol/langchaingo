package ollama

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/vxcontrol/langchaingo/llms"
)

func sentOptions(t *testing.T, client []Option, opts ...llms.CallOption) map[string]any {
	t.Helper()

	raw, err := sendChatRequestWithClient(t, "glm-5", client, opts...)
	require.NoError(t, err)
	var body map[string]any
	require.NoError(t, json.Unmarshal(raw, &body))
	options, _ := body["options"].(map[string]any)
	return options
}

func TestSamplingReachesTheWireInTheDoorsOwnShape(t *testing.T) {
	t.Parallel()

	body := captureChatRequest(t,
		llms.WithTemperature(0.3),
		llms.WithTopP(0.7),
		llms.WithTopK(11),
		llms.WithMaxTokens(555),
		llms.WithSeed(7),
		llms.WithStopWords([]string{"STOP"}),
		llms.WithRepetitionPenalty(1.3),
		llms.WithFrequencyPenalty(0.4),
		llms.WithPresencePenalty(0.6),
	)

	options, ok := body["options"].(map[string]any)
	require.True(t, ok, "no options in the request; body=%v", body)

	for field, want := range map[string]float64{
		"temperature":       0.3,
		"top_p":             0.7,
		"top_k":             11,
		"num_predict":       555,
		"seed":              7,
		"repeat_penalty":    1.3,
		"frequency_penalty": 0.4,
		"presence_penalty":  0.6,
	} {
		got, present := options[field]
		require.Truef(t, present, "%s never reached the wire; options=%v", field, options)
		require.InDeltaf(t, want, got, 1e-6, "%s = %v, want %v", field, got, want)
	}

	require.Equal(t, []any{"STOP"}, options["stop"], "stop words did not reach the wire")
}

func TestAZeroTheCallerAsksForReachesTheWire(t *testing.T) {
	t.Parallel()

	options := sentOptions(t, nil, llms.WithTemperature(0), llms.WithSeed(0), llms.WithTopP(0), llms.WithTopK(0),
		llms.WithMinP(0), llms.WithRepetitionPenalty(0))
	for _, field := range []string{"temperature", "seed", "top_p", "top_k", "min_p", "repeat_penalty"} {
		got, present := options[field]
		require.Truef(t, present, "%s 0 was asked and the server would use its own default; options=%v", field, options)
		require.InDeltaf(t, 0, got, 0, field)
	}
}

func TestTheClientsSamplingReachesTheWireUnlessTheCallSetsIt(t *testing.T) {
	t.Parallel()

	client := []Option{WithSeed(42), WithTopK(9), WithTopP(0.5), WithMinP(0.1), WithNumPredict(77)}

	options := sentOptions(t, client)
	for field, want := range map[string]float64{"seed": 42, "top_k": 9, "top_p": 0.5, "min_p": 0.1, "num_predict": 77} {
		require.InDeltaf(t, want, options[field], 1e-6, "%s set on the client; options=%v", field, options)
	}

	options = sentOptions(t, client, llms.WithSeed(1), llms.WithTopK(3), llms.WithMinP(0.2), llms.WithMaxTokens(5))
	for field, want := range map[string]float64{"seed": 1, "top_k": 3, "top_p": 0.5, "min_p": 0.2, "num_predict": 5} {
		require.InDeltaf(t, want, options[field], 1e-6, "%s: the call wins over the client; options=%v", field, options)
	}
}

func TestSamplingNobodySetStaysOffTheWire(t *testing.T) {
	t.Parallel()

	options := sentOptions(t, nil)
	for _, field := range []string{"temperature", "seed", "top_p", "top_k", "min_p", "repeat_penalty", "stop"} {
		require.NotContainsf(t, options, field, "the server's default applies; options=%v", options)
	}
	require.InDelta(t, llms.DefaultMaxTokens, options["num_predict"], 0)
}
