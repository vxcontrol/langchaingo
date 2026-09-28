package jina

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProbeVfy3Jina(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		data := []map[string]any{}
		for i := range req.Input {
			data = append(data, map[string]any{"object": "embedding", "index": i, "embedding": []float32{0.1, 0.2}})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"model": "x", "object": "list", "data": data, "usage": map[string]any{"total_tokens": 1, "prompt_tokens": 1}})
	}))
	defer srv.Close()
	for _, bs := range []int{0, -1} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("bs=%d PANIC: %v\n", bs, r)
				}
			}()
			j, _ := NewJina(WithAPIBaseURL(srv.URL), WithAPIKey("k"), WithBatchSize(bs))
			fmt.Printf("bs=%d BatchSize=%d\n", bs, j.BatchSize)
			embs, err := j.EmbedDocuments(context.Background(), []string{"a", "b"})
			fmt.Printf("bs=%d embs=%d err=%v\n", bs, len(embs), err)
		}()
	}
}
