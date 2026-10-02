package bedrock_test

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestNovaIsSentATemperatureFromTheFloorToOne(t *testing.T) {
	t.Parallel()

	novaTemperature := func(body map[string]any) any {
		cfg, _ := body["inferenceConfig"].(map[string]any)
		return cfg["temperature"]
	}
	for _, tc := range []struct {
		model  string
		answer string
		opts   []bedrock.Option
	}{
		{"us.amazon.nova-pro-v1:0", novaAnswer, nil},
		{"amazon.nova-micro-v1:0", novaAnswer, nil},
		{"us.amazon.nova-lite-v1:0", converseAnswer, []bedrock.Option{bedrock.WithConverseAPI()}},
		{"arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-premier-v1:0", converseAnswer,
			[]bedrock.Option{bedrock.WithConverseAPI()}},
		{"us.amazon.nova-2-lite-v1:0", novaAnswer, nil},
		{"amazon.nova-2-lite-v1:0", converseAnswer, []bedrock.Option{bedrock.WithConverseAPI()}},
	} {
		for _, c := range []struct {
			asked, sent float64
			reported    bool
		}{
			{asked: 0, sent: 0.00001, reported: true},
			{asked: 1.5, sent: 1, reported: true},
			{asked: 0.5, sent: 0.5},
			{asked: 1, sent: 1},
		} {
			opts := append([]bedrock.Option{bedrock.WithModel(tc.model)}, tc.opts...)
			resp, body := bedrockWarningsSending(t, tc.answer, opts, llms.WithTemperature(c.asked))

			sent := novaTemperature(body)
			require.NotNil(t, sent, "%s asked %v: no temperature on the wire", tc.model, c.asked)
			require.InDelta(t, c.sent, sent, 1e-9, "%s asked %v", tc.model, c.asked)
			w, ok := bedrockWarningsByOption(resp.Warnings)["WithTemperature"]
			if !c.reported {
				require.False(t, ok, "%s asked %v: %v", tc.model, c.asked, resp.Warnings)
				continue
			}
			require.True(t, ok, "%s asked %v: the clamp went unreported: %v", tc.model, c.asked, resp.Warnings)
			require.Equal(t, llms.WarningClamp, w.Kind)
			require.Equal(t, strconv.FormatFloat(c.asked, 'g', -1, 64), w.Asked)
			require.Equal(t, strconv.FormatFloat(c.sent, 'g', -1, 64), w.Sent)
		}
	}
}
