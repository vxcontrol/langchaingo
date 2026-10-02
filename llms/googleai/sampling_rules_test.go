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

func generationConfigSent(t *testing.T, model string, opts ...llms.CallOption) (map[string]any, map[string]llms.Warning) {
	t.Helper()

	var body []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{}}`)
	}))
	t.Cleanup(server.Close)

	llm, err := New(context.Background(),
		WithAPIKey("unit-test-key"), WithEndpoint(server.URL), WithDefaultModel(model))
	require.NoError(t, err)

	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	require.NoError(t, err)

	var sent struct {
		GenerationConfig map[string]any `json:"generationConfig"`
	}
	require.NoError(t, json.Unmarshal(body, &sent))
	warnings := map[string]llms.Warning{}
	for _, w := range resp.Warnings {
		warnings[w.Option] = w
	}
	return sent.GenerationConfig, warnings
}

func TestAGemini3ModelIsSentNoCandidateCount(t *testing.T) {
	t.Parallel()

	for _, model := range []string{
		"gemini-3.8-flash", "gemini-3-flash-preview", "gemini-4-flash", "gemini-flash-latest", "models/gemini-3.8-flash",
	} {
		config, warnings := generationConfigSent(t, model, llms.WithCandidateCount(2))
		assert.NotContains(t, config, "candidateCount", model)
		if assert.Contains(t, warnings, "WithCandidateCount", model) {
			assert.Equal(t, llms.WarningDrop, warnings["WithCandidateCount"].Kind, model)
			assert.Equal(t, "2", warnings["WithCandidateCount"].Asked, model)
		}

		config, warnings = generationConfigSent(t, model)
		assert.NotContains(t, config, "candidateCount", model)
		assert.NotContains(t, warnings, "WithCandidateCount", model, "the door's own default is not the caller's request")
	}

	config, warnings := generationConfigSent(t, "gemini-2.5-flash",
		llms.WithModel("gemini-3.8-flash"), llms.WithCandidateCount(2))
	assert.NotContains(t, config, "candidateCount", "the per-call model decides, not the client's default")
	assert.Contains(t, warnings, "WithCandidateCount")

	for _, model := range []string{"gemini-2.5-flash", "gemini-2.5-flash-native-audio-latest"} {
		config, warnings := generationConfigSent(t, model, llms.WithCandidateCount(2))
		assert.InDelta(t, 2, config["candidateCount"], 0, model)
		assert.NotContains(t, warnings, "WithCandidateCount", model)
	}
}

func TestVertexIsSentNoCandidateCountForAGemini3Model(t *testing.T) {
	t.Parallel()

	_, body, resp := vertexCall(t, llms.WithModel("gemini-3.8-flash"), llms.WithCandidateCount(2))
	config, _ := body["generationConfig"].(map[string]any)
	assert.NotContains(t, config, "candidateCount")
	dropped, reported := googleWarningsByOption(resp.Warnings)["WithCandidateCount"]
	if assert.True(t, reported, "%v", resp.Warnings) {
		assert.Equal(t, llms.WarningDrop, dropped.Kind)
		assert.Equal(t, "2", dropped.Asked)
	}

	_, body, resp = vertexCall(t, llms.WithModel("gemini-3.8-flash"))
	config, _ = body["generationConfig"].(map[string]any)
	assert.NotContains(t, config, "candidateCount")
	assert.NotContains(t, googleWarningsByOption(resp.Warnings), "WithCandidateCount",
		"the door's own default is not the caller's request")

	_, body, _ = vertexCall(t, llms.WithCandidateCount(2))
	config, _ = body["generationConfig"].(map[string]any)
	assert.InDelta(t, 2, config["candidateCount"], 0, "Gemini 2.5 on Vertex keeps the count")
}
