package bedrock_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

func TestKimiK3IsSentAHistoryWithoutTheReasoningOfEarlierTurns(t *testing.T) {
	t.Parallel()

	assistantBlocks := func(model string) []string {
		var raw []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			raw, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, converseAnswer)
		}))
		t.Cleanup(srv.Close)

		llm := bedrockLLMAgainst(t, srv, bedrock.WithModel(model), bedrock.WithConverseAPI())
		_, err := llm.GenerateContent(t.Context(), []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "add two and two"),
			{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("4",
				&reasoning.ContentReasoning{Content: "two plus two", Signature: []byte("sig")})}},
			llms.TextParts(llms.ChatMessageTypeHuman, "and three more?"),
		})
		require.NoError(t, err, model)

		var sent struct {
			Messages []struct {
				Role    string           `json:"role"`
				Content []map[string]any `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.Unmarshal(raw, &sent), model)
		var blocks []string
		for _, message := range sent.Messages {
			if message.Role != "assistant" {
				continue
			}
			for _, block := range message.Content {
				for kind := range block {
					blocks = append(blocks, kind)
				}
			}
		}
		return blocks
	}

	for _, model := range []string{"moonshotai.kimi-k3", "us.moonshotai.kimi-k3", "global.moonshotai.kimi-k3"} {
		require.Equal(t, []string{"text"}, assistantBlocks(model), model)
	}
	require.Equal(t, []string{"reasoningContent", "text"}, assistantBlocks("moonshotai.kimi-k2.5"),
		"other models keep the reasoning they were given")
}

func TestKimiK3IsNotSentTwoUserTurnsWhereAnAnswerWasOnlyReasoning(t *testing.T) {
	t.Parallel()

	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, converseAnswer)
	}))
	t.Cleanup(srv.Close)

	llm := bedrockLLMAgainst(t, srv, bedrock.WithModel("moonshotai.kimi-k3"), bedrock.WithConverseAPI())
	_, err := llm.GenerateContent(t.Context(), []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "add two and two"),
		{Role: llms.ChatMessageTypeAI, Parts: []llms.ContentPart{llms.TextPartWithReasoning("",
			&reasoning.ContentReasoning{Content: "two plus two", Signature: []byte("sig")})}},
		llms.TextParts(llms.ChatMessageTypeHuman, "go on"),
	})
	require.NoError(t, err)

	var sent struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(raw, &sent))
	roles := make([]string, 0, len(sent.Messages))
	for _, message := range sent.Messages {
		roles = append(roles, message.Role)
	}
	require.Equal(t, []string{"user"}, roles, "the emptied turn merges the user turns around it")
}
