package openai

import (
	"cmp"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/httputil"
	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms"
)

const (
	xaiHostURL      = "http://api.x.ai/v1"
	moonshotHostURL = "http://api.moonshot.ai/v1"
	mistralHostURL  = "http://api.mistral.ai/v1"
)

type vendorModelAPI struct {
	mu      sync.Mutex
	answers map[string]string
	lookups []string
	body    map[string]any
}

func (a *vendorModelAPI) llm(t *testing.T, baseURL, model string, opts ...Option) *LLM {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		defer a.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			a.lookups = append(a.lookups, r.URL.Path)
			answer, ok := a.answers[r.URL.Path]
			if !ok {
				http.NotFound(w, r)
				return
			}
			_, _ = io.WriteString(w, answer)
			return
		}
		a.body = nil
		_ = json.NewDecoder(r.Body).Decode(&a.body)
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","created":1,"model":"m",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(append([]Option{
		WithBaseURL(baseURL), WithToken("token"), WithModel(model), WithHTTPClient(clientDialing(t, srv)),
	}, opts...)...)
	require.NoError(t, err)
	return llm
}

func (a *vendorModelAPI) send(
	t *testing.T, llm *LLM, messages []llms.MessageContent, opts ...llms.CallOption,
) (map[string]any, map[string]llms.Warning, error) {
	t.Helper()

	resp, err := llm.GenerateContent(context.Background(), messages, opts...)
	warnings := map[string]llms.Warning{}
	if resp != nil {
		for _, w := range resp.Warnings {
			warnings[w.Option] = w
		}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.body, warnings, err
}

func (a *vendorModelAPI) ask(t *testing.T, llm *LLM, opts ...llms.CallOption) (map[string]any, map[string]llms.Warning) {
	t.Helper()

	body, warnings, err := a.send(t, llm, []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
	require.NoError(t, err)
	return body, warnings
}

func (a *vendorModelAPI) asked() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.lookups
}

const grokReport = `{"id":"grok-4.8","object":"model","owned_by":"xai","aliases":[],` +
	`"capabilities":{"reasoning_effort":["high","low"],"default_reasoning_effort":"high"}}`

func TestAGrokModelTheTablesDoNotListTakesTheEffortsXAIReports(t *testing.T) {
	t.Parallel()

	api := &vendorModelAPI{answers: map[string]string{"/v1/models/grok-4.8": grokReport}}
	llm := api.llm(t, xaiHostURL, "grok-4.8")
	for range 2 {
		body, warnings := api.ask(t, llm, llms.WithReasoning(llms.ReasoningMedium, 0))
		assert.Equal(t, "low", body["reasoning_effort"], "grok-4.7, whose rules grok-4.8 inherits, takes medium")
		assert.Equal(t, llms.WarningClamp, warnings["WithReasoning"].Kind)
		assert.Contains(t, warnings["WithReasoning"].Reason, "the API reports")
		assert.Contains(t, warnings["WithModel"].Reason, "what the API reports")

		body, _ = api.ask(t, llm, llms.WithReasoning(llms.ReasoningXHigh, 0))
		assert.Equal(t, "high", body["reasoning_effort"])
	}
	assert.Equal(t, []string{"/v1/models/grok-4.8"}, api.asked(), "the model is asked about once per TTL")

	api = &vendorModelAPI{}
	body, warnings := api.ask(t, api.llm(t, xaiHostURL, "grok-4.8"), llms.WithReasoning(llms.ReasoningMedium, 0))
	assert.Equal(t, "medium", body["reasoning_effort"], "without the endpoint the inherited rules decide")
	assert.NotContains(t, warnings["WithModel"].Reason, "API reports")
	assert.Len(t, api.asked(), 1)
}

func TestAGrokModelXAIReportsWithoutEffortsGetsNoEffortField(t *testing.T) {
	t.Parallel()

	api := &vendorModelAPI{answers: map[string]string{
		"/v1/models/grok-4.8": `{"id":"grok-4.8","aliases":[],"capabilities":{"reasoning_effort":[]}}`,
	}}
	body, warnings := api.ask(t, api.llm(t, xaiHostURL, "grok-4.8"), llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.NotContains(t, body, "reasoning_effort")
	assert.Equal(t, llms.WarningDrop, warnings["WithReasoning"].Kind)
	assert.Contains(t, warnings["WithReasoning"].Reason, "the API reports no effort values")

	api = &vendorModelAPI{answers: map[string]string{
		"/v1/models/grok-4.8": `{"id":"grok-4.8","aliases":[],"capabilities":{"reasoning_effort":["none"]}}`,
	}}
	body, _ = api.ask(t, api.llm(t, xaiHostURL, "grok-4.8"), llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.NotContains(t, body, "reasoning_effort", "none is the disable token, not an effort")

	api = &vendorModelAPI{answers: map[string]string{
		"/v1/models/grok-4.8": `{"id":"grok-4.8","aliases":[],"capabilities":null}`,
	}}
	body, _ = api.ask(t, api.llm(t, xaiHostURL, "grok-4.8"), llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.Equal(t, "high", body["reasoning_effort"], "a model without capabilities reports nothing")
}

func TestXAIIsNotAskedWhereTheTablesDecide(t *testing.T) {
	t.Parallel()

	high := llms.WithReasoning(llms.ReasoningHigh, 0)
	for name, call := range map[string]struct {
		baseURL, model string
		opts           []Option
		call           []llms.CallOption
	}{
		"a documented release":         {xaiHostURL, "grok-4.7", nil, []llms.CallOption{high}},
		"a release without effort row": {xaiHostURL, "grok-4.3", nil, []llms.CallOption{high}},
		"a model without effort":       {xaiHostURL, "grok-4.20-0309-reasoning", nil, []llms.CallOption{high}},
		"a gateway route":              {gatewayBaseURL, "xai/grok-4.8", nil, []llms.CallOption{high}},
		"the lookup switched off":      {xaiHostURL, "grok-4.8", []Option{WithoutModelLookup()}, []llms.CallOption{high}},
		"a call without reasoning":     {xaiHostURL, "grok-4.8", nil, nil},
		"thinking switched off":        {xaiHostURL, "grok-4.8", nil, []llms.CallOption{llms.WithReasoningDisabled()}},
		"xAI's report has no thinking": {xaiHostURL, "grok-4.8", nil, []llms.CallOption{llms.WithTemperature(0.5)}},
	} {
		api := &vendorModelAPI{answers: map[string]string{"/v1/models/grok-4.8": grokReport}}
		api.ask(t, api.llm(t, call.baseURL, call.model, call.opts...), call.call...)
		assert.Empty(t, api.asked(), name)
	}
}

const moonshotListing = `{"object":"list","data":[` +
	`{"id":"kimi-k3","object":"model","owned_by":"moonshot","context_length":262144,"supports_reasoning":true},` +
	`{"id":"kimi-k3.5","object":"model","owned_by":"moonshot","context_length":262144,"supports_reasoning":false},` +
	`{"id":"moonshot-v1-8k","object":"model","owned_by":"moonshot","context_length":8192,"supports_reasoning":true}]}`

func TestAKimiModelMoonshotReportsAsNotThinkingIsSentNoThinking(t *testing.T) {
	t.Parallel()

	api := &vendorModelAPI{answers: map[string]string{"/v1/models": moonshotListing}}
	llm := api.llm(t, moonshotHostURL, "kimi-k3.5")
	body, warnings := api.ask(t, llm, llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.NotContains(t, body, "reasoning_effort", "kimi-k3, whose rules kimi-k3.5 inherits, takes high")
	assert.NotContains(t, body, "thinking")
	assert.Equal(t, llms.WarningDrop, warnings["WithReasoning"].Kind)
	assert.Equal(t, "the API reports that the model does not think", warnings["WithReasoning"].Reason)
	assert.Contains(t, warnings["WithModel"].Reason, "what the API reports")

	body, warnings = api.ask(t, llm, llms.WithReasoningDisabled())
	assert.NotContains(t, body, "reasoning_effort")
	assert.NotContains(t, body, "thinking")
	assert.NotContains(t, warnings, "WithReasoningDisabled", "a model that does not think needs no off switch")
	assert.Equal(t, []string{"/v1/models"}, api.asked())

	api = &vendorModelAPI{}
	_, warnings = api.ask(t, api.llm(t, moonshotHostURL, "kimi-k3.5"), llms.WithReasoningDisabled())
	assert.Equal(t, llms.WarningInherit, warnings["WithReasoningDisabled"].Kind, "kimi-k3 cannot switch thinking off")
	body, _ = api.ask(t, api.llm(t, moonshotHostURL, "kimi-k3.5"), llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.Equal(t, "high", body["reasoning_effort"])
}

func TestAKimiModelThatDoesNotThinkCanBeForcedToCallATool(t *testing.T) {
	t.Parallel()

	forced := []llms.CallOption{
		llms.WithTools([]llms.Tool{astraTool()}),
		llms.WithToolChoice(llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: astraTool().Function.Name}}),
	}
	messages := []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "look it up")}

	api := &vendorModelAPI{answers: map[string]string{"/v1/models": moonshotListing}}
	body, _, err := api.send(t, api.llm(t, moonshotHostURL, "kimi-k3.5"), messages, forced...)
	require.NoError(t, err)
	assert.NotEmpty(t, body["tool_choice"])

	api = &vendorModelAPI{}
	_, _, err = api.send(t, api.llm(t, moonshotHostURL, "kimi-k3.5"), messages, forced...)
	require.Error(t, err, "Moonshot refuses a forced tool while a Kimi model thinks")
}

func TestMoonshotIsNotAskedWhereTheTablesDecide(t *testing.T) {
	t.Parallel()

	for name, call := range map[string]struct{ baseURL, model string }{
		"a documented release": {moonshotHostURL, "kimi-k3"},
		"a gateway route":      {gatewayBaseURL, "moonshot/kimi-k3.5"},
	} {
		api := &vendorModelAPI{answers: map[string]string{"/v1/models": moonshotListing}}
		api.ask(t, api.llm(t, call.baseURL, call.model), llms.WithReasoning(llms.ReasoningHigh, 0))
		assert.Empty(t, api.asked(), name)
	}

	api := &vendorModelAPI{answers: map[string]string{"/v1/models": `{"object":"list","data":[{"id":"kimi-k3"}]}`}}
	body, _ := api.ask(t, api.llm(t, moonshotHostURL, "kimi-k3.5"), llms.WithReasoning(llms.ReasoningHigh, 0))
	assert.Equal(t, "high", body["reasoning_effort"], "a model missing from the listing follows the tables")
	assert.Len(t, api.asked(), 1)

	api = &vendorModelAPI{answers: map[string]string{"/v1/models": moonshotListing}}
	body, _ = api.ask(t, api.llm(t, moonshotHostURL, "moonshot-v1-8k"), llms.WithReasoningDisabled())
	assert.NotContains(t, body, "reasoning_effort", "Mistral's disable token is not Moonshot's")
	assert.Len(t, api.asked(), 1)
}

func mistralCard(id string, reasons bool) string {
	card, _ := json.Marshal(map[string]any{
		"id": id, "object": "model", "type": "base", "aliases": []string{},
		"capabilities": map[string]bool{"completion_chat": true, "function_calling": true, "reasoning": reasons},
	})
	return string(card)
}

func TestAMistralModelTheAPIReportsAsReasoningIsSentReasoningEffort(t *testing.T) {
	t.Parallel()

	answers := map[string]string{"/v1/models/mistral-small-2610": mistralCard("mistral-small-2610", true)}
	api := &vendorModelAPI{answers: answers}
	llm := api.llm(t, mistralHostURL, "mistral-small-2610", WithPreserveReasoningContent())
	body, warnings := api.ask(t, llm, llms.WithReasoning(llms.ReasoningMedium, 0))
	assert.Equal(t, "high", body["reasoning_effort"])
	assert.Equal(t, llms.WarningClamp, warnings["WithReasoning"].Kind)

	body, _ = api.ask(t, llm, llms.WithReasoningDisabled())
	assert.Equal(t, "none", body["reasoning_effort"])

	body, _, err := api.send(t, llm, replayHistory())
	require.NoError(t, err)
	messages, _ := body["messages"].([]any)
	require.Len(t, messages, 5)
	turn, _ := messages[1].(map[string]any)
	assert.Equal(t, []any{
		map[string]any{"type": "thinking", "thinking": []any{map[string]any{"type": "text", "text": "text turn thought"}}},
		map[string]any{"type": "text", "text": "answered in text"},
	}, turn["content"], "the model's own thinking goes back as Mistral takes it")
	assert.Equal(t, []string{"/v1/models/mistral-small-2610"}, api.asked())

	api = &vendorModelAPI{answers: map[string]string{
		"/v1/models/mistral-small-2610": mistralCard("mistral-small-2610", false),
	}}
	llm = api.llm(t, mistralHostURL, "mistral-small-2610")
	body, warnings = api.ask(t, llm, llms.WithReasoning(llms.ReasoningMedium, 0))
	assert.NotContains(t, body, "reasoning_effort")
	assert.Equal(t, llms.WarningDrop, warnings["WithReasoning"].Kind)
	body, _ = api.ask(t, llm, llms.WithReasoningDisabled())
	assert.NotContains(t, body, "reasoning_effort")

	api = &vendorModelAPI{}
	body, _ = api.ask(t, api.llm(t, mistralHostURL, "mistral-small-2610"), llms.WithReasoning(llms.ReasoningMedium, 0))
	assert.NotContains(t, body, "reasoning_effort", "the tables take an unlisted Mistral model for one that does not reason")
}

func TestMistralIsNotAskedWhereTheTablesDecide(t *testing.T) {
	t.Parallel()

	for name, call := range map[string]struct{ baseURL, model string }{
		"a documented reasoning model": {mistralHostURL, "mistral-small-latest"},
		"GLM on Mistral":               {mistralHostURL, "zai-glm-5-3"},
		"a gateway route":              {gatewayBaseURL, "mistral/mistral-small-2610"},
	} {
		api := &vendorModelAPI{answers: map[string]string{
			"/v1/models/" + call.model: mistralCard(call.model, false),
		}}
		api.ask(t, api.llm(t, call.baseURL, call.model), llms.WithReasoning(llms.ReasoningHigh, 0))
		assert.Empty(t, api.asked(), name)
	}

	api := &vendorModelAPI{answers: map[string]string{
		"/v1/models/mistral-small-2610": mistralCard("mistral-small-2610", true),
	}}
	llm := api.llm(t, mistralHostURL, "mistral-small-2610", WithPreserveReasoningContent())
	api.ask(t, llm)
	_, _, err := api.send(t, llm, []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "first"),
		llms.TextParts(llms.ChatMessageTypeAI, "answered without thinking"),
		llms.TextParts(llms.ChatMessageTypeHuman, "second"),
	})
	require.NoError(t, err)
	assert.Empty(t, api.asked(), "a call that asks nothing of reasoning and replays no thinking decides nothing the report changes")
}

func TestMistralsModelCardIsReadForReasoning(t *testing.T) {
	httprr.SkipIfNoCredentialsAndRecordingMissing(t, "MISTRAL_API_KEY")
	transport := httputil.DefaultTransport
	if base := os.Getenv("MISTRAL_BASE_URL"); base != "" {
		transport = &httputil.ApiKeyTransport{Transport: transport, BaseURL: base}
	}
	rr := httprr.OpenForTest(t, transport)
	if rr.Replaying() {
		t.Parallel()
	}
	token := "test-api-key"
	if !rr.Replaying() {
		token = cmp.Or(os.Getenv("MISTRAL_API_KEY"), token)
	}
	llm, err := New(WithBaseURL("https://api.mistral.ai/v1"), WithToken(token), WithHTTPClient(rr.Client()))
	require.NoError(t, err)

	for model, reasons := range map[string]bool{"mistral-small-latest": true, "codestral-latest": false} {
		report, err := llm.mistralReport(t.Context(), model)
		require.NoError(t, err)
		require.NotNil(t, report.thinks, model)
		assert.Equal(t, reasons, *report.thinks, model)
	}
}
