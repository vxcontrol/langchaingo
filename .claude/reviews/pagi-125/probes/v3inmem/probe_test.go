package inmemory_test

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"testing"

	"github.com/vxcontrol/langchaingo/schema"
	"github.com/vxcontrol/langchaingo/vectorstores/inmemory"
)

type v3emb struct{}

func vec(s string) []float32 {
	if s == "q" {
		return []float32{1, 0, 0}
	}
	i, _ := strconv.Atoi(s[1:])
	a := float64(i) * 0.09
	return []float32{float32(math.Cos(a)), float32(math.Sin(a)), float32(0.1 * float64(i%3))}
}
func (v3emb) EmbedDocuments(_ context.Context, ts []string) ([][]float32, error) {
	out := make([][]float32, len(ts))
	for i, t := range ts {
		out[i] = vec(t)
	}
	return out, nil
}
func (v3emb) EmbedQuery(_ context.Context, t string) ([]float32, error) { return vec(t), nil }

func TestProbeV3Inmem(t *testing.T) {
	ctx := context.Background()
	st, err := inmemory.New(ctx, inmemory.WithEmbedder(v3emb{}), inmemory.WithVectorSize(3))
	if err != nil {
		t.Fatal(err)
	}
	var docs []schema.Document
	for i := 0; i < 20; i++ {
		docs = append(docs, schema.Document{PageContent: fmt.Sprintf("d%d", i)})
	}
	if _, err := st.AddDocuments(ctx, docs); err != nil {
		t.Fatal(err)
	}
	res, err := st.SimilaritySearch(ctx, "q", 8)
	if err != nil {
		t.Fatal(err)
	}
	sorted := true
	s := ""
	for i, d := range res {
		if i > 0 && d.Score > res[i-1].Score {
			sorted = false
		}
		s += fmt.Sprintf("%s:%.4f ", d.PageContent, d.Score)
	}
	t.Logf("sorted=%v %s", sorted, s)
}
