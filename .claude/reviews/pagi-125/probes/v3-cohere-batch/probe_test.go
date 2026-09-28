package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

func TestProbeCohereBatch(t *testing.T) {
	var sizes []int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var in struct {
			Texts []string `json:"texts"`
		}
		_ = json.Unmarshal(b, &in)
		sizes = append(sizes, len(in.Texts))
		if len(in.Texts) > 96 {
			w.Header().Set("X-Amzn-Errortype", "ValidationException")
			w.WriteHeader(400)
			fmt.Fprint(w, `{"message":"Malformed input request: #/texts: expected maximum item count: 96"}`)
			return
		}
		embs := make([][]float32, len(in.Texts))
		for i := range embs {
			embs[i] = []float32{1}
		}
		json.NewEncoder(w).Encode(map[string]any{"response_type": "embeddings_floats", "embeddings": embs})
	}))
	defer srv.Close()
	cl := bedrockruntime.New(bedrockruntime.Options{Region: "us-east-1", BaseEndpoint: aws.String(srv.URL), Credentials: credentials.NewStaticCredentialsProvider("AK", "SK", ""), AuthSchemePreference: []string{"sigv4"}})
	e, err := NewBedrock(WithModel(ModelCohereEn), WithClient(cl))
	if err != nil {
		t.Fatal(err)
	}
	texts := make([]string, 200)
	for i := range texts {
		texts[i] = fmt.Sprint("t", i)
	}
	v, err := e.EmbedDocuments(context.Background(), texts)
	t.Logf("BatchSize=%d request sizes=%v got=%d err=%v", e.BatchSize, sizes, len(v), err)
}
