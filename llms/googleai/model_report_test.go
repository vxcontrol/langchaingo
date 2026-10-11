package googleai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

type geminiModelAPI struct {
	mu       sync.Mutex
	model    string
	vertex   bool
	lookups  []string
	requests []map[string]any
}

func (m *geminiModelAPI) serve(t *testing.T, defaultModel string, opts ...Option) *GoogleAI {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.mu.Lock()
		defer m.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/models/") {
			m.lookups = append(m.lookups, r.URL.Path)
			if m.model == "" {
				w.WriteHeader(http.StatusNotFound)
				_, _ = io.WriteString(w, `{"error":{"code":404,"message":"not found","status":"NOT_FOUND"}}`)
				return
			}
			_, _ = io.WriteString(w, m.model)
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		m.requests = append(m.requests, body)
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{}}`)
	}))
	t.Cleanup(srv.Close)

	backend := []Option{WithAPIKey("unit-test-key"), WithEndpoint(srv.URL)}
	if m.vertex {
		backend = []Option{
			WithCloudProject("hotel-desk"), WithCloudLocation("europe-west4"),
			WithHTTPClient(&http.Client{Transport: &toLocalServer{host: srv.Listener.Addr().String()}}),
		}
	}
	llm, err := New(context.Background(), append(append(backend, WithDefaultModel(defaultModel)), opts...)...)
	require.NoError(t, err)
	return llm
}

func (m *geminiModelAPI) generationConfig(t *testing.T) map[string]any {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	require.NotEmpty(t, m.requests, "no generateContent request reached the API")
	config, _ := m.requests[len(m.requests)-1]["generationConfig"].(map[string]any)
	return config
}

const geminiModelRecord = `{"name":"models/gemini-3.9-flash","inputTokenLimit":1048576,"outputTokenLimit":65536,` +
	`"supportedGenerationMethods":["generateContent"],"temperature":1,"maxTemperature":2,"topP":0.95,"topK":64}`

func geminiRecord(t *testing.T, fields map[string]any) string {
	t.Helper()
	var record map[string]any
	require.NoError(t, json.Unmarshal([]byte(geminiModelRecord), &record))
	for k, v := range fields {
		if v == nil {
			delete(record, k)
			continue
		}
		record[k] = v
	}
	data, err := json.Marshal(record)
	require.NoError(t, err)
	return string(data)
}

func askGemini(t *testing.T, llm *GoogleAI, opts ...llms.CallOption) *llms.ContentResponse {
	t.Helper()
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	require.NoError(t, err)
	return resp
}

func TestAGeminiModelTheAPIReportsAsNotThinkingIsSentNoThinking(t *testing.T) {
	t.Parallel()

	api := &geminiModelAPI{model: geminiRecord(t, map[string]any{"thinking": false})}
	llm := api.serve(t, "gemini-3.9-flash")

	resp := askGemini(t, llm, llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.NotContains(t, api.generationConfig(t), "thinkingConfig",
		"the inherited Gemini 3.8 Flash row thinks; the API reports that this model does not")
	dropped, ok := googleWarningsByOption(resp.Warnings)["WithReasoning"]
	require.True(t, ok, "%v", resp.Warnings)
	assert.Equal(t, llms.WarningDrop, dropped.Kind)
	assert.Contains(t, googleWarningsByOption(resp.Warnings)["WithModel"].Reason, "what the API reports")

	resp = askGemini(t, llm, llms.WithReasoningDisabled())
	assert.NotContains(t, api.generationConfig(t), "thinkingConfig", "a model that does not think needs no off switch")
	assert.NotContains(t, googleWarningsByOption(resp.Warnings), "WithReasoningDisabled")
	assert.NotContains(t, googleWarningsByOption(resp.Warnings), "WithReasoning")
	assert.Len(t, api.lookups, 1, "the model is asked about once per TTL")
}

func TestAGeminiModelThatDoesNotThinkIsNotRefusedAnOffSwitch(t *testing.T) {
	t.Parallel()

	api := &geminiModelAPI{}
	resp := askGemini(t, api.serve(t, "gemini-3.9-pro"), llms.WithReasoningDisabled())
	refused, ok := googleWarningsByOption(resp.Warnings)["WithReasoningDisabled"]
	require.True(t, ok, "Gemini 3.1 Pro cannot switch thinking off: %v", resp.Warnings)
	assert.Equal(t, llms.WarningInherit, refused.Kind)

	api = &geminiModelAPI{model: geminiRecord(t, map[string]any{"name": "models/gemini-3.9-pro", "thinking": false})}
	resp = askGemini(t, api.serve(t, "gemini-3.9-pro"), llms.WithReasoningDisabled())
	assert.NotContains(t, api.generationConfig(t), "thinkingConfig")
	assert.NotContains(t, googleWarningsByOption(resp.Warnings), "WithReasoningDisabled")
}

func TestAGeminiModelTheAPIReportsAsThinkingKeepsTheTablesThinking(t *testing.T) {
	t.Parallel()

	api := &geminiModelAPI{model: geminiRecord(t, map[string]any{"thinking": true})}
	askGemini(t, api.serve(t, "gemini-3.9-flash"), llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.Contains(t, api.generationConfig(t), "thinkingConfig")
	assert.Len(t, api.lookups, 1)
}

func TestTopKIsDroppedWhereTheAPIReportsNoTopKSampling(t *testing.T) {
	t.Parallel()

	api := &geminiModelAPI{model: geminiRecord(t, map[string]any{"thinking": true, "topK": nil})}
	resp := askGemini(t, api.serve(t, "gemini-3.9-flash"), llms.WithTopK(40))
	assert.NotContains(t, api.generationConfig(t), "topK")
	dropped, ok := googleWarningsByOption(resp.Warnings)["WithTopK"]
	require.True(t, ok, "%v", resp.Warnings)
	assert.Equal(t, llms.WarningDrop, dropped.Kind)
	assert.Equal(t, "40", dropped.Asked)

	api = &geminiModelAPI{model: geminiRecord(t, map[string]any{"thinking": true})}
	askGemini(t, api.serve(t, "gemini-3.9-flash"), llms.WithTopK(40))
	assert.InDelta(t, 40, api.generationConfig(t)["topK"], 0, "a model that reports a topK default takes topK")
}

func TestATemperatureAboveTheReportedMaximumIsClamped(t *testing.T) {
	t.Parallel()

	api := &geminiModelAPI{model: geminiRecord(t, map[string]any{"thinking": true, "maxTemperature": 1.5})}
	resp := askGemini(t, api.serve(t, "gemini-3.9-flash"), llms.WithTemperature(1.8))
	assert.InDelta(t, 1.5, api.generationConfig(t)["temperature"], 1e-6)
	clamped, ok := googleWarningsByOption(resp.Warnings)["WithTemperature"]
	require.True(t, ok, "%v", resp.Warnings)
	assert.Equal(t, llms.WarningClamp, clamped.Kind)
	assert.Equal(t, "1.8", clamped.Asked)
	assert.Equal(t, "1.5", clamped.Sent)
}

func TestTheGeminiAPIIsNotAskedWhereTheTablesDecide(t *testing.T) {
	t.Parallel()

	api := &geminiModelAPI{model: geminiRecord(t, map[string]any{"thinking": false, "topK": nil})}
	askGemini(t, api.serve(t, "gemini-3.8-flash"), llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithTopK(40))
	assert.Empty(t, api.lookups, "a documented model follows its table row")
	assert.Contains(t, api.generationConfig(t), "thinkingConfig")

	api = &geminiModelAPI{model: geminiRecord(t, map[string]any{"thinking": false})}
	askGemini(t, api.serve(t, "gemini-3.9-flash", WithoutModelLookup()), llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.Empty(t, api.lookups)
	assert.Contains(t, api.generationConfig(t), "thinkingConfig")

	api = &geminiModelAPI{model: geminiRecord(t, map[string]any{"thinking": false}), vertex: true}
	askGemini(t, api.serve(t, "gemini-3.9-flash"), llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.Empty(t, api.lookups, "the SDK reads no capability fields from a Vertex AI model")
	assert.Contains(t, api.generationConfig(t), "thinkingConfig")

	api = &geminiModelAPI{model: geminiRecord(t, map[string]any{"thinking": false})}
	askGemini(t, api.serve(t, "gemini-3.9-flash"))
	assert.Empty(t, api.lookups, "a call that asks for no thinking and no sampling decides nothing the report changes")
}

func TestAGeminiHostWithoutTheModelEndpointLeavesTheModelToTheTables(t *testing.T) {
	t.Parallel()

	api := &geminiModelAPI{}
	llm := api.serve(t, "gemini-3.9-flash")
	for range 2 {
		resp := askGemini(t, llm, llms.WithReasoning(llms.ReasoningHigh, 0))
		assert.Contains(t, api.generationConfig(t), "thinkingConfig")
		assert.NotContains(t, googleWarningsByOption(resp.Warnings)["WithModel"].Reason, "API reports")
	}
	assert.Len(t, api.lookups, 1)

	api = &geminiModelAPI{model: `{"id":"gemini-3.9-flash","object":"model","owned_by":"google"}`}
	askGemini(t, api.serve(t, "gemini-3.9-flash"), llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.Len(t, api.lookups, 1)
	assert.Contains(t, api.generationConfig(t), "thinkingConfig", "an answer that is not a Gemini model record reports nothing")
}

func TestTheGeminiReportIsReadFromTheVendorsModelAnswer(t *testing.T) {
	llm := newHTTPRRClient(t)

	m, err := llm.client.Models.Get(t.Context(), "gemini-3.8-flash", nil)
	require.NoError(t, err)
	assert.Equal(t, geminiReport{reported: true, thinking: true, topK: 64, maxTemperature: 2}, geminiReportFrom(m))
}
