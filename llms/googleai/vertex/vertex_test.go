package vertex_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/googleai"
	"github.com/vxcontrol/langchaingo/llms/googleai/vertex"
)

type toLocalServer struct {
	host string
	seen *string
}

func (t *toLocalServer) RoundTrip(req *http.Request) (*http.Response, error) {
	*t.seen = req.URL.String()
	req.URL.Scheme = "http"
	req.URL.Host = t.host
	return http.DefaultTransport.RoundTrip(req)
}

func vertexAgainstALocalServer(t *testing.T, body string, opts ...googleai.Option) (*vertex.Vertex, *string) {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)

	seen := new(string)
	llm, err := vertex.New(context.Background(), append([]googleai.Option{
		googleai.WithCloudProject("hotel-desk"),
		googleai.WithCloudLocation("europe-west4"),
		googleai.WithDefaultModel("gemini-2.5-flash"),
		googleai.WithHTTPClient(&http.Client{
			Transport: &toLocalServer{host: srv.Listener.Addr().String(), seen: seen},
		}),
	}, opts...)...)
	require.NoError(t, err)
	return llm, seen
}

func TestVertexRefusesToGuessWhereItIs(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")

	_, err := vertex.New(context.Background())
	require.ErrorIs(t, err, vertex.ErrMissingCloudTarget,
		"a door that reaches one cloud must be told which project and location, not fall back to another backend")
}

func TestVertexTakesItsCloudTargetFromTheEnvironment(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "night-porter")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`)
	}))
	t.Cleanup(srv.Close)

	seen := new(string)
	llm, err := vertex.New(context.Background(),
		googleai.WithDefaultModel("gemini-2.5-flash"),
		googleai.WithHTTPClient(&http.Client{
			Transport: &toLocalServer{host: srv.Listener.Addr().String(), seen: seen},
		}))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	require.NoError(t, err)

	assert.Contains(t, *seen, "/projects/night-porter/locations/us-central1/",
		"the environment names the cloud target when the caller does not")
}

func TestVertexAddressesTheVertexBackend(t *testing.T) {
	t.Parallel()

	llm, seen := vertexAgainstALocalServer(t,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"sixty rooms are free"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":5,"totalTokenCount":8}}`)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")})
	require.NoError(t, err)

	assert.Contains(t, *seen, "-aiplatform.googleapis.com/",
		"the door must reach Vertex, not the Gemini API")
	assert.Contains(t, *seen, "/projects/hotel-desk/locations/europe-west4/")
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
	assert.Equal(t, 3, resp.Choices[0].GenerationInfo["PromptTokens"])
}

func TestVertexEmbedsThroughTheSameBackend(t *testing.T) {
	t.Parallel()

	llm, seen := vertexAgainstALocalServer(t,
		`{"predictions":[{"embeddings":{"values":[0.1,0.2]}}],`+
			`"embeddings":[{"values":[0.1,0.2]}]}`)

	embeddings, err := llm.CreateEmbedding(context.Background(), []string{"a room"})
	require.NoError(t, err)
	require.Len(t, embeddings, 1)
	assert.Equal(t, []float32{0.1, 0.2}, embeddings[0])
	assert.Contains(t, *seen, "-aiplatform.googleapis.com/",
		"embeddings go to the same backend as generation, with no second client")
}

func vertexRequestBody(t *testing.T, call ...llms.CallOption) map[string]any {
	t.Helper()

	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`)
	}))
	t.Cleanup(srv.Close)

	seen := new(string)
	llm, err := vertex.New(context.Background(),
		googleai.WithCloudProject("hotel-desk"), googleai.WithCloudLocation("europe-west4"),
		googleai.WithDefaultModel("gemini-2.5-flash"),
		googleai.WithHTTPClient(&http.Client{
			Transport: &toLocalServer{host: srv.Listener.Addr().String(), seen: seen},
		}))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, call...)
	require.NoError(t, err)

	var body map[string]any
	require.NoError(t, json.Unmarshal([]byte(got), &body))
	return body
}

func TestVertexCarriesTheThinkingBudgetTheCallerAskedFor(t *testing.T) {
	t.Parallel()

	body := vertexRequestBody(t, llms.WithReasoning(llms.ReasoningNone, 2048))

	config, ok := body["generationConfig"].(map[string]any)
	require.True(t, ok)
	thinking, ok := config["thinkingConfig"].(map[string]any)
	require.True(t, ok, "the door used to accept a thinking request and send nothing")
	assert.Equal(t, float64(2048), thinking["thinkingBudget"])
}

func TestVertexCarriesTheToolChoiceTheCallerAskedFor(t *testing.T) {
	t.Parallel()

	tools := []llms.Tool{{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "rooms_free",
			Description: "how many rooms are free",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"floor": map[string]any{"type": "integer"}},
			},
		},
	}}

	body := vertexRequestBody(t, llms.WithTools(tools), llms.WithToolChoice("required"))

	config, ok := body["toolConfig"].(map[string]any)
	require.True(t, ok, "the door used to convert the tools and drop the choice")
	calling, ok := config["functionCallingConfig"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "ANY", calling["mode"])
}
