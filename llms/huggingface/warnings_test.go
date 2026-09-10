package huggingface

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func generateForWarnings(t *testing.T, messages []llms.MessageContent, call ...llms.CallOption) *llms.ContentResponse {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithToken("t"), WithURL(srv.URL), WithModel("Qwen/Qwen3-32B"))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(context.Background(), messages, call...)
	require.NoError(t, err)
	return resp
}

func hfWarningsByOption(warnings []llms.Warning) map[string]llms.Warning {
	byOption := make(map[string]llms.Warning, len(warnings))
	for _, w := range warnings {
		byOption[w.Option] = w
	}
	return byOption
}

func oneMessage() []llms.MessageContent {
	return []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
}

func TestEveryOptionWithNoFieldOnThisDoorIsReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, oneMessage(),
		llms.WithMaxTokens(100), llms.WithTopK(5), llms.WithMinLength(2),
		llms.WithN(2), llms.WithCandidateCount(3), llms.WithTopLogProbs(4),
		llms.WithMinP(0.1), llms.WithRepetitionPenalty(1.1),
		llms.WithFrequencyPenalty(0.5), llms.WithPresencePenalty(0.25),
		llms.WithLogProbs(true), llms.WithStopWords([]string{"STOP"}),
		llms.WithJSONMode(),
	)

	got := hfWarningsByOption(resp.Warnings)
	for option, asked := range map[string]string{
		"WithMaxTokens": "100", "WithTopK": "5", "WithMinLength": "2",
		"WithN": "2", "WithCandidateCount": "3", "WithTopLogProbs": "4",
		"WithMinP": "0.1", "WithRepetitionPenalty": "1.1",
		"WithFrequencyPenalty": "0.5", "WithPresencePenalty": "0.25",
		"WithLogProbs": "true", "WithStopWords": "1 words", "WithJSONMode": "true",
	} {
		w, ok := got[option]
		require.True(t, ok, "no %s warning in %v", option, resp.Warnings)
		require.Equal(t, llms.WarningDrop, w.Kind)
		require.Equal(t, asked, w.Asked)
		require.Equal(t, "Qwen/Qwen3-32B", w.Model)
	}
}

func TestTheMessagesThisDoorLeavesBehindAreReported(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "be brief"),
		llms.TextParts(llms.ChatMessageTypeHuman, "hi", "and again"),
	})

	w, ok := hfWarningsByOption(resp.Warnings)["messages"]
	require.True(t, ok, "no messages warning in %v", resp.Warnings)
	require.Equal(t, llms.WarningClamp, w.Kind)
	require.Equal(t, "3 parts", w.Asked)
	require.Equal(t, "1 part", w.Sent)
}

func TestAPlainHuggingFaceCallCarriesNoWarnings(t *testing.T) {
	t.Parallel()

	resp := generateForWarnings(t, oneMessage(), llms.WithTemperature(0.2))
	require.Empty(t, resp.Warnings)
}
