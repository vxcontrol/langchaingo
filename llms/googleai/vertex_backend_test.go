package googleai

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
)

func vertexCall(t *testing.T, call ...llms.CallOption) (string, map[string]any, *llms.ContentResponse) {
	t.Helper()

	var gotURL, gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotURL, gotBody = r.URL.String(), string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"sixty rooms are free"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":5,"totalTokenCount":8}}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(context.Background(),
		WithCloudProject("hotel-desk"),
		WithCloudLocation("europe-west4"),
		WithDefaultModel("gemini-2.5-flash"),
		WithHTTPClient(&http.Client{Transport: &toLocalServer{host: srv.Listener.Addr().String()}}))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")}, call...)
	require.NoError(t, err)

	var body map[string]any
	require.NoError(t, json.Unmarshal([]byte(gotBody), &body))
	return gotURL, body, resp
}

type toLocalServer struct {
	host string
}

func (t *toLocalServer) RoundTrip(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = "http"
	req.URL.Host = t.host
	return http.DefaultTransport.RoundTrip(req)
}

func TestTheVertexBackendAddressesTheProjectTheCallerNamed(t *testing.T) {
	t.Parallel()

	got, _, resp := vertexCall(t)

	assert.Contains(t, got, "/projects/hotel-desk/locations/europe-west4/",
		"the project and location the caller gave decide the address")
	assert.Contains(t, got, "/publishers/google/models/gemini-2.5-flash:generateContent",
		"the model reaches the vendor's own path shape")

	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "sixty rooms are free", resp.Choices[0].Content)
	assert.Equal(t, 3, resp.Choices[0].GenerationInfo["PromptTokens"])
	assert.Equal(t, 5, resp.Choices[0].GenerationInfo["CompletionTokens"])
}

func TestTheVertexBackendCarriesAThinkingBudget(t *testing.T) {
	t.Parallel()

	_, body, _ := vertexCall(t, llms.WithReasoning(llms.ReasoningNone, 2048))

	config, ok := body["generationConfig"].(map[string]any)
	require.True(t, ok, "the request carries a generation config")
	thinking, ok := config["thinkingConfig"].(map[string]any)
	require.True(t, ok, "a caller that asked for a budget gets a thinking config on the wire")
	assert.Equal(t, float64(2048), thinking["thinkingBudget"],
		"the budget the caller named is the budget the vendor is told")
}

func TestTheVertexBackendCarriesTheToolChoice(t *testing.T) {
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

	_, body, _ := vertexCall(t, llms.WithTools(tools), llms.WithToolChoice("required"))

	config, ok := body["toolConfig"].(map[string]any)
	require.True(t, ok, "a caller that required a tool gets a tool config on the wire")
	calling, ok := config["functionCallingConfig"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "ANY", calling["mode"],
		"required means the vendor's ANY, not the default AUTO")
}

func TestTheVertexBackendCarriesBothPenalties(t *testing.T) {
	t.Parallel()

	_, body, _ := vertexCall(t,
		llms.WithFrequencyPenalty(0.7), llms.WithPresencePenalty(0.4))

	config, ok := body["generationConfig"].(map[string]any)
	require.True(t, ok, "the request carries a generation config")
	assert.InDelta(t, 0.7, config["frequencyPenalty"], 1e-6,
		"the frequency penalty the caller named reaches the vendor")
	assert.InDelta(t, 0.4, config["presencePenalty"], 1e-6,
		"the presence penalty the caller named reaches the vendor")
}
