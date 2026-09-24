package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

func thinkingOnlyStream(t *testing.T) *LLM {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, thought := range []string{"counting the ", "free rooms"} {
			_, _ = io.WriteString(w, `{"model":"llama3","created_at":"2026-08-21T09:00:00Z",`+
				`"message":{"role":"assistant","content":"","thinking":"`+thought+`"},"done":false}`+"\n")
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel("llama3"))
	require.NoError(t, err)
	return llm
}

func TestAReasoningOnlyStreamIsNotThrownAway(t *testing.T) {
	t.Parallel()

	llm := thinkingOnlyStream(t)

	gaveUp := errors.New("consumer gave up")
	delivered := 0
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
			if chunk.Type != streaming.ChunkTypeReasoning {
				return nil
			}
			delivered++
			if delivered == 2 {
				return gaveUp
			}
			return nil
		}))

	require.ErrorIs(t, err, gaveUp)
	require.NotNil(t, resp, "a stream that produced only reasoning still produced something")
	require.Len(t, resp.Choices, 1)
	require.NotNil(t, resp.Choices[0].Reasoning)
	assert.Contains(t, resp.Choices[0].Reasoning.Content, "counting the ")
}

func TestABrokenStreamStillCarriesTheThinkingItDelivered(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = io.WriteString(w, `{"model":"llama3","created_at":"2026-08-21T09:00:00Z",`+
			`"message":{"role":"assistant","content":"","thinking":"counting the free rooms"},"done":false}`+"\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		_, _ = io.WriteString(w, "{not json at all\n")
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel("llama3"))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")},
		llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil }))

	require.Error(t, err, "an undecodable line is not a finished answer")
	require.NotNil(t, resp, "handleChat assembles thinking even when the message carries no content")
	require.Len(t, resp.Choices, 1)
	require.NotNil(t, resp.Choices[0].Reasoning)
	assert.Equal(t, "counting the free rooms", resp.Choices[0].Reasoning.Content)
}

// cutServer writes the given NDJSON lines and closes the response cleanly,
// without the final frame (done: true) that normally ends an answer. Ollama up
// to 0.34.0 ends a stream this way when it stops a model repeating itself.
func cutServer(t *testing.T, lines ...string) *LLM {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, line := range lines {
			_, _ = io.WriteString(w, line+"\n")
		}
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel("llama3"))
	require.NoError(t, err)
	return llm
}

func cutFrame(content string) string {
	frame, _ := json.Marshal(map[string]any{
		"model": "llama3", "message": map[string]any{"role": "assistant", "content": content}, "done": false,
	})
	return string(frame)
}

func TestAStreamThatEndsBeforeItsFinalFrameKeepsItsText(t *testing.T) {
	t.Parallel()

	human := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "question")}
	schema := llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "answer", Schema: json.RawMessage(ollamaSOSchema)})
	ignore := llms.WithStreamingFunc(func(context.Context, streaming.Chunk) error { return nil })

	t.Run("streaming keeps the text instead of an empty answer", func(t *testing.T) {
		t.Parallel()

		var streamed strings.Builder
		resp, err := cutServer(t, cutFrame("Paris is "), cutFrame("the capital.")).GenerateContent(t.Context(), human,
			llms.WithStreamingFunc(func(_ context.Context, chunk streaming.Chunk) error {
				if chunk.Type == streaming.ChunkTypeText {
					streamed.WriteString(chunk.Content)
				}
				return nil
			}))
		require.NoError(t, err)
		assert.Equal(t, "Paris is the capital.", resp.Choices[0].Content)
		assert.Equal(t, "Paris is the capital.", streamed.String())
	})

	t.Run("a schema is judged on the text that arrived", func(t *testing.T) {
		t.Parallel()

		resp, err := cutServer(t, cutFrame(`{"answer":`), cutFrame(`"42"}`)).GenerateContent(t.Context(), human, schema, ignore)
		require.NoError(t, err)
		assert.Equal(t, `{"answer":"42"}`, resp.Choices[0].Content)

		resp, err = cutServer(t, cutFrame(`{"answer":`), cutFrame(`"4`)).GenerateContent(t.Context(), human, schema, ignore)
		var validation *llms.ErrStructuredOutputValidation
		require.ErrorAs(t, err, &validation)
		assert.Equal(t, `{"answer":"4`, resp.Choices[0].Content)
	})

	t.Run("a non-streaming answer without its final frame stays a success", func(t *testing.T) {
		t.Parallel()

		resp, err := cutServer(t, cutFrame("Paris")).GenerateContent(t.Context(), human)
		require.NoError(t, err)
		assert.Equal(t, "Paris", resp.Choices[0].Content)

		out, err := llms.GenerateFromSinglePrompt(t.Context(), cutServer(t, cutFrame("Paris")), "question")
		require.NoError(t, err)
		assert.Equal(t, "Paris", out)
	})

	t.Run("nothing at all is still an empty answer", func(t *testing.T) {
		t.Parallel()

		resp, err := cutServer(t).GenerateContent(t.Context(), human, ignore)
		require.NoError(t, err)
		require.NotNil(t, resp)
		assert.Empty(t, resp.Choices[0].Content)
	})
}
