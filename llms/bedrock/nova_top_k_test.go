package bedrock_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestNovaCarriesTopKWhereTheNovaSchemaPutsItOnBothDoors(t *testing.T) {
	t.Parallel()

	topKSent := func(body map[string]any, converse bool) (any, bool) {
		section := body["inferenceConfig"]
		if converse {
			fields, _ := body["additionalModelRequestFields"].(map[string]any)
			section = fields["inferenceConfig"]
		}
		config, _ := section.(map[string]any)
		value, ok := config["topK"]
		return value, ok
	}
	for _, door := range []struct {
		name     string
		converse bool
		answer   string
	}{
		{"converse", true, converseAnswer},
		{"legacy", false, novaAnswer},
	} {
		t.Run(door.name, func(t *testing.T) {
			t.Parallel()

			call := func(model string, opts ...llms.CallOption) (*llms.ContentResponse, map[string]any) {
				bedrockOpts := []bedrock.Option{bedrock.WithModel(model)}
				if door.converse {
					bedrockOpts = append(bedrockOpts, bedrock.WithConverseAPI())
				}
				return bedrockWarningsSending(t, door.answer, bedrockOpts, opts...)
			}

			for _, model := range []string{"us.amazon.nova-pro-v1:0", "amazon.nova-2-lite-v1:0"} {
				resp, body := call(model, llms.WithTopK(40))
				value, sent := topKSent(body, door.converse)
				require.True(t, sent, "%s: %v", model, body)
				require.InDelta(t, 40, value, 0, model)
				require.NotContains(t, bedrockWarningsByOption(resp.Warnings), "WithTopK", model)
			}

			for _, model := range []string{"us.amazon.nova-pro-v1:0", "amazon.nova-2-lite-v1:0"} {
				resp, body := call(model, llms.WithTopK(500))
				value, sent := topKSent(body, door.converse)
				require.True(t, sent, "%s: %v", model, body)
				require.InDelta(t, 128, value, 0, "%s: the Nova schema takes a topK from 0 to 128", model)
				clamped, reported := bedrockWarningsByOption(resp.Warnings)["WithTopK"]
				require.True(t, reported, "%s: %v", model, resp.Warnings)
				require.Equal(t, llms.WarningClamp, clamped.Kind)
				require.Equal(t, "500", clamped.Asked)
				require.Equal(t, "128", clamped.Sent)

				resp, body = call(model, llms.WithTopK(128))
				value, _ = topKSent(body, door.converse)
				require.InDelta(t, 128, value, 0, model)
				require.NotContains(t, bedrockWarningsByOption(resp.Warnings), "WithTopK", model)
			}

			_, body := call("amazon.nova-2-lite-v1:0", llms.WithTopK(40), llms.WithReasoning(llms.ReasoningLow, 0))
			value, sent := topKSent(body, door.converse)
			require.True(t, sent, "a low effort leaves the sampling in place: %v", body)
			require.InDelta(t, 40, value, 0)

			resp, body := call("amazon.nova-2-lite-v1:0", llms.WithTopK(40), llms.WithReasoning(llms.ReasoningHigh, 0))
			_, sent = topKSent(body, door.converse)
			require.False(t, sent, "Nova refuses topK at the top effort: %v", body)
			dropped, reported := bedrockWarningsByOption(resp.Warnings)["WithTopK"]
			require.True(t, reported, "%v", resp.Warnings)
			require.Equal(t, llms.WarningDrop, dropped.Kind)
			require.Equal(t, "40", dropped.Asked)
		})
	}
}
