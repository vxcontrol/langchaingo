package inmemory_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores/inmemory"
)

type probeEmb struct{}

func (probeEmb) vec(s string) []float32 {
	var n int
	fmt.Sscanf(s, "d%d", &n)
	return []float32{1, float32(n) / 10, 0.3}
}
func (p probeEmb) EmbedDocuments(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = p.vec(t)
	}
	return out, nil
}
func (p probeEmb) EmbedQuery(_ context.Context, _ string) ([]float32, error) {
	return []float32{1, 0, 0.3}, nil
}

func TestProbeTrcOrder(t *testing.T) {
	ctx := context.Background()
	s, err := inmemory.New(ctx, inmemory.WithEmbedder(probeEmb{}), inmemory.WithVectorSize(3))
	if err != nil {
		t.Fatal(err)
	}
	var docs []schema.Document
	for i := 0; i < 20; i++ {
		docs = append(docs, schema.Document{PageContent: fmt.Sprintf("d%d", (i*7)%20)})
	}
	if _, err := s.AddDocuments(ctx, docs); err != nil {
		t.Fatal(err)
	}
	res, err := s.SimilaritySearch(ctx, "q", 8)
	if err != nil {
		t.Fatal(err)
	}
	sorted := true
	for i := 1; i < len(res); i++ {
		if res[i].Score > res[i-1].Score {
			sorted = false
		}
	}
	var out []string
	for _, d := range res {
		out = append(out, fmt.Sprintf("%s:%.4f", d.PageContent, d.Score))
	}
	t.Logf("sorted=%v %v", sorted, out)
}
