package jina

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProbeJinaZeroBatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req EmbeddingRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		resp := map[string]any{"data": func() []any {
			out := []any{}
			for i := range req.Input {
				out = append(out, map[string]any{"index": i, "embedding": []float32{1, 2}})
			}
			return out
		}()}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()
	j, _ := NewJina(WithAPIBaseURL(srv.URL), WithAPIKey("k"), WithBatchSize(0))
	t.Logf("BatchSize=%d", j.BatchSize)
	defer func() {
		if r := recover(); r != nil {
			t.Logf("PANIC: %v", r)
		}
	}()
	embs, err := j.EmbedDocuments(context.Background(), []string{"a", "b"})
	t.Logf("embs=%d err=%v", len(embs), err)
}
