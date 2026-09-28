package pinecone

import (
	"testing"

	"github.com/pinecone-io/go-pinecone/v4/pinecone"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestProbeVerPineOrder(t *testing.T) {
	store := Store{textKey: "text"}
	m := func(text string, score float32) *pinecone.ScoredVector {
		md, _ := structpb.NewStruct(map[string]any{"text": text})
		return &pinecone.ScoredVector{Vector: &pinecone.Vector{Metadata: md}, Score: score}
	}
	// Pinecone's own order for a euclidean index: nearest (smallest distance) first.
	docs, err := store.getDocumentsFromMatches(&pinecone.QueryVectorsResponse{Matches: []*pinecone.ScoredVector{
		m("nearest", 0.05), m("middle", 0.40), m("farthest", 1.20),
	}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i, d := range docs {
		t.Logf("euclidean %d %s %.2f", i, d.PageContent, d.Score)
	}
}
