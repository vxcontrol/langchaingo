package anthropic_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/anthropic"
)

func TestProbeLegacyUnknownEffortHits(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"completion":"ok","stop_reason":"stop_sequence"}`)
	}))
	defer srv.Close()
	llm, _ := anthropic.New(anthropic.WithToken("k"), anthropic.WithBaseURL(srv.URL),
		anthropic.WithModel("claude-2.1"), anthropic.WithLegacyTextCompletionsAPI())
	_, err := llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
		llms.WithReasoning("enormous", 0))
	t.Logf("err=%v hits=%d", err, hits.Load())
}
