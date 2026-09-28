package pinecone

import (
	"testing"

	"github.com/pinecone-io/go-pinecone/v4/pinecone"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestProbeEuclideanOrder(t *testing.T) {
	store := Store{textKey: "text"}
	match := func(text string, score float32) *pinecone.ScoredVector {
		md, _ := structpb.NewStruct(map[string]any{"text": text})
		return &pinecone.ScoredVector{Vector: &pinecone.Vector{Metadata: md}, Score: score}
	}
	// Order and scores as a euclidean index returns them: smallest distance first.
	docs, err := store.getDocumentsFromMatches(&pinecone.QueryVectorsResponse{Matches: []*pinecone.ScoredVector{
		match("nearest", 0.05), match("middle", 0.40), match("farthest", 1.20),
	}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i, d := range docs {
		t.Logf("%d %s %.2f", i, d.PageContent, d.Score)
	}
}
