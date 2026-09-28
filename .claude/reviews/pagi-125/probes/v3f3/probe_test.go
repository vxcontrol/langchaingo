package pinecone

import (
	"testing"

	"github.com/pinecone-io/go-pinecone/v4/pinecone"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestProbeV3F3(t *testing.T) {
	store := Store{textKey: "text"}
	m := func(text string, s float32) *pinecone.ScoredVector {
		md, _ := structpb.NewStruct(map[string]any{"text": text})
		return &pinecone.ScoredVector{Vector: &pinecone.Vector{Metadata: md}, Score: s}
	}
	docs, err := store.getDocumentsFromMatches(&pinecone.QueryVectorsResponse{Matches: []*pinecone.ScoredVector{
		m("nearest", 0.05), m("middle", 0.4), m("farthest", 1.7)}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range docs {
		t.Logf("%s %v", d.PageContent, d.Score)
	}
}
