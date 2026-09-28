package huggingface

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// A mirror of router.huggingface.co that serves hf-inference feature extraction
// on both the documented migration path (/hf-inference/models/{model}) and the
// pipeline path, and 404s elsewhere.
func TestProbeHFEmbeddingsWithHFInferenceBaseURL(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		got = r.URL.Path
		switch r.URL.Path {
		case "/hf-inference/models/BAAI/bge-small-en-v1.5",
			"/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `[[0.1,0.2]]`)
		default:
			w.WriteHeader(http.StatusNotFound)
			_, _ = io.WriteString(w, `Not Found`)
		}
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithToken("t"), WithURL(srv.URL+"/hf-inference"))
	if err != nil {
		t.Fatal(err)
	}
	emb, err := llm.CreateEmbedding(context.Background(), []string{"hi"}, "BAAI/bge-small-en-v1.5", "feature-extraction")
	fmt.Printf("WithURL(<router>/hf-inference): path=%s err=%v n=%d\n", got, err, len(emb))
}
