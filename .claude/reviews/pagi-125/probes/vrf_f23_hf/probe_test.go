package huggingface

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProbeVrfHF(t *testing.T) {
	var paths []string
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		paths = append(paths, r.URL.Path)
		bodies = append(bodies, string(b))
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path[len(r.URL.Path)-len("completions"):] == "completions" {
			io.WriteString(w, `{"choices":[{"message":{"content":"ok"},"index":0,"finish_reason":"stop"}]}`)
			return
		}
		io.WriteString(w, `[[0.1]]`)
	}))
	defer srv.Close()
	llm, err := New(WithToken("t"), WithURL(srv.URL), WithInferenceProvider("scaleway"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = llm.CreateEmbedding(context.Background(), []string{"hi"}, "Qwen/Qwen3-Embedding-8B", "feature-extraction")
	t.Logf("scaleway embed err=%v path=%v body=%v", err, paths, bodies)
	paths, bodies = nil, nil
	llm2, err := New(WithToken("t"), WithURL(srv.URL), WithInferenceProvider("groq"), WithModel("meta-llama/Llama-3.3-70B-Instruct"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = llm2.Call(context.Background(), "hi")
	t.Logf("groq chat err=%v path=%v", err, paths)
}
