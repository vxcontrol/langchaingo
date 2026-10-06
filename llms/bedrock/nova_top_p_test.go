package bedrock_test

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestNovaIsSentATopPFromTheFloorToOneOnBothDoors(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		model  string
		answer string
		opts   []bedrock.Option
	}{
		{"us.amazon.nova-pro-v1:0", novaAnswer, nil},
		{"us.amazon.nova-pro-v1:0", converseAnswer, []bedrock.Option{bedrock.WithConverseAPI()}},
		{"amazon.nova-2-lite-v1:0", novaAnswer, nil},
		{"amazon.nova-2-lite-v1:0", converseAnswer, []bedrock.Option{bedrock.WithConverseAPI()}},
	} {
		for _, c := range []struct {
			asked, sent float64
			reported    bool
		}{
			{asked: 0, sent: 0.00001, reported: true},
			{asked: 1.3, sent: 1, reported: true},
			{asked: 0.9, sent: 0.9},
			{asked: 1, sent: 1},
		} {
			opts := append([]bedrock.Option{bedrock.WithModel(tc.model)}, tc.opts...)
			resp, body := bedrockWarningsSending(t, tc.answer, opts, llms.WithTopP(c.asked))

			cfg, _ := body["inferenceConfig"].(map[string]any)
			require.InDelta(t, c.sent, cfg["topP"], 1e-6, "%s %v asked %v: %v", tc.model, tc.opts, c.asked, body)
			w, ok := bedrockWarningsByOption(resp.Warnings)["WithTopP"]
			if !c.reported {
				require.False(t, ok, "%s %v asked %v: %v", tc.model, tc.opts, c.asked, resp.Warnings)
				continue
			}
			require.True(t, ok, "%s %v asked %v: the clamp went unreported: %v", tc.model, tc.opts, c.asked, resp.Warnings)
			require.Equal(t, llms.WarningClamp, w.Kind)
			require.Equal(t, strconv.FormatFloat(c.asked, 'g', -1, 64), w.Asked)
			require.Equal(t, strconv.FormatFloat(c.sent, 'g', -1, 64), w.Sent)
			require.Equal(t, "Nova takes a topP from 0.00001 to 1", w.Reason)
		}
	}
}
