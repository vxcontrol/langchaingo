package huggingface_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	hfemb "github.com/vxcontrol/langchaingo/embeddings/huggingface"
	hf "github.com/vxcontrol/langchaingo/llms/huggingface"
)

func TestProbeVrfxHFEmbedBase(t *testing.T) {
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		got = append(got, r.URL.Path)
		switch r.URL.Path {
		case "/hf-inference/models/BAAI/bge-small-en-v1.5",
			"/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `[[0.1,0.2]]`)
		default:
			w.WriteHeader(404)
			_, _ = io.WriteString(w, "Not Found")
		}
	}))
	defer srv.Close()
	for _, base := range []string{srv.URL + "/hf-inference", srv.URL} {
		got = nil
		llm, err := hf.New(hf.WithToken("t"), hf.WithURL(base))
		if err != nil {
			t.Fatal(err)
		}
		e, err := hfemb.NewHuggingface(hfemb.WithClient(*llm))
		if err != nil {
			t.Fatal(err)
		}
		v, err := e.EmbedQuery(context.Background(), "hi")
		fmt.Printf("base=%q paths=%v err=%v len=%d\n", base[len(srv.URL):], got, err, len(v))
	}
}
