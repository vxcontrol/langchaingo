package bedrock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func bedrockWarningsFor(t *testing.T, answer string, opts []bedrock.Option, call ...llms.CallOption) *llms.ContentResponse {
	t.Helper()

	resp, _ := bedrockWarningsSending(t, answer, opts, call...)
	return resp
}

func bedrockWarningsSending(
	t *testing.T, answer string, opts []bedrock.Option, call ...llms.CallOption,
) (*llms.ContentResponse, map[string]any) {
	t.Helper()

	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, answer)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv, opts...)
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, call...)
	require.NoError(t, err)
	var sent map[string]any
	require.NoError(t, json.Unmarshal(body, &sent))
	return resp, sent
}

func bedrockWarningsByOption(warnings []llms.Warning) map[string]llms.Warning {
	byOption := make(map[string]llms.Warning, len(warnings))
	for _, w := range warnings {
		byOption[w.Option] = w
	}
	return byOption
}

func TestOptionsNeitherBedrockRequestCarriesAreReported(t *testing.T) {
	t.Parallel()

	call := []llms.CallOption{
		llms.WithMaxTokens(1024),
		llms.WithMinP(0.05), llms.WithRepetitionPenalty(1.1),
		llms.WithFrequencyPenalty(0.3), llms.WithPresencePenalty(0.7),
		llms.WithN(2), llms.WithCandidateCount(3),
		llms.WithLogProbs(true), llms.WithTopLogProbs(5),
	}
	want := map[string]string{
		"WithMinP": "0.05", "WithRepetitionPenalty": "1.1",
		"WithFrequencyPenalty": "0.3", "WithPresencePenalty": "0.7",
		"WithN": "2", "WithCandidateCount": "3",
		"WithLogProbs": "true", "WithTopLogProbs": "5",
	}

	for _, door := range []struct {
		name   string
		answer string
		opts   []bedrock.Option
	}{
		{"legacy", legacyAnswer, []bedrock.Option{bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0")}},
		{"converse", converseAnswer, []bedrock.Option{
			bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0"), bedrock.WithConverseAPI(),
		}},
	} {
		t.Run(door.name, func(t *testing.T) {
			t.Parallel()

			got := bedrockWarningsByOption(bedrockWarningsFor(t, door.answer, door.opts, call...).Warnings)
			for option, asked := range want {
				w, ok := got[option]
				require.True(t, ok, "no %s warning on the %s door", option, door.name)
				require.Equal(t, llms.WarningDrop, w.Kind)
				require.Equal(t, asked, w.Asked)
			}
		})
	}
}

func TestABedrockCallThatSetsNothingExtraCarriesNoSuchWarnings(t *testing.T) {
	t.Parallel()

	resp := bedrockWarningsFor(t, legacyAnswer,
		[]bedrock.Option{bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0")},
		llms.WithMaxTokens(1024))

	require.Empty(t, resp.Warnings)
}

func TestAPayloadThatCarriesThePenaltiesReportsNoLoss(t *testing.T) {
	t.Parallel()

	const ai21Answer = `{"completions":[{"data":{"text":"ok"},"finishReason":{"reason":"endoftext"}}]}`

	resp, sent := bedrockWarningsSending(t, ai21Answer,
		[]bedrock.Option{bedrock.WithModel("ai21.j2-ultra-v1")},
		llms.WithRepetitionPenalty(1.1), llms.WithFrequencyPenalty(0.3),
		llms.WithPresencePenalty(0.7), llms.WithCandidateCount(3),
		llms.WithMinP(0.05),
	)

	got := bedrockWarningsByOption(resp.Warnings)
	for _, option := range []string{
		"WithRepetitionPenalty", "WithFrequencyPenalty", "WithPresencePenalty", "WithCandidateCount",
	} {
		require.NotContains(t, got, option, "the ai21 payload carries it: %v", resp.Warnings)
	}
	require.Contains(t, got, "WithMinP", "no payload carries min-p")

	for field, want := range map[string]any{
		"countPenalty":     map[string]any{"scale": 1.1},
		"frequencyPenalty": map[string]any{"scale": 0.3},
		"presencePenalty":  map[string]any{"scale": 0.7},
		"numResults":       float64(3),
	} {
		require.Equal(t, want, sent[field], "%s on the ai21 payload", field)
	}
}

func TestTheConversePathReportsWhatTheLegacyPayloadWouldHaveCarried(t *testing.T) {
	t.Parallel()

	resp := bedrockWarningsFor(t, converseAnswer,
		[]bedrock.Option{bedrock.WithModel("ai21.jamba-1-5-large-v1:0"), bedrock.WithConverseAPI()},
		llms.WithFrequencyPenalty(0.3), llms.WithCandidateCount(3),
	)

	got := bedrockWarningsByOption(resp.Warnings)
	require.Contains(t, got, "WithFrequencyPenalty", "ConverseInput has no penalty field")
	require.Contains(t, got, "WithCandidateCount", "ConverseInput has no candidate-count field")
}

func TestAskingForJSONOnBedrockIsReported(t *testing.T) {
	t.Parallel()

	resp := bedrockWarningsFor(t, legacyAnswer,
		[]bedrock.Option{bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0")},
		llms.WithJSONMode())

	w, ok := bedrockWarningsByOption(resp.Warnings)["WithJSONMode"]
	require.True(t, ok, "no json-mode warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningDrop, w.Kind)
}

func TestExtraBodyNeitherBedrockDoorCanMergeIsReported(t *testing.T) {
	t.Parallel()

	for _, door := range []struct {
		name   string
		answer string
		opts   []bedrock.Option
	}{
		{"legacy", legacyAnswer, []bedrock.Option{bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0")}},
		{"converse", converseAnswer, []bedrock.Option{
			bedrock.WithModel("anthropic.claude-sonnet-4-5-v1:0"), bedrock.WithConverseAPI(),
		}},
	} {
		t.Run(door.name, func(t *testing.T) {
			t.Parallel()

			got := bedrockWarningsByOption(bedrockWarningsFor(t, door.answer, door.opts,
				llms.WithExtraBody(map[string]any{"enable_thinking": false, "chat_template_kwargs": map[string]any{}})).Warnings)
			w, ok := got["WithExtraBody"]
			require.True(t, ok, "no extra-body warning on the %s door", door.name)
			require.Equal(t, llms.WarningDrop, w.Kind)
			require.Equal(t, "chat_template_kwargs, enable_thinking", w.Asked)
		})
	}
}
