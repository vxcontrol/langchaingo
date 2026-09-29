package ollama

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/reasoning"
)

type showServer struct {
	shows atomic.Int32
	chats atomic.Int32
	body  atomic.Value
}

func newShowServer(t *testing.T, show string) (*showServer, *LLM) {
	t.Helper()

	s := &showServer{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/api/show":
			s.shows.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, show)
		case "/api/chat":
			s.chats.Add(1)
			s.body.Store(body)
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = io.WriteString(w, `{"model":"m","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`+"\n")
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel("m"))
	require.NoError(t, err)
	return s, llm
}

func (s *showServer) sentThink(t *testing.T) (any, bool) {
	t.Helper()

	var got map[string]any
	require.NoError(t, json.Unmarshal(s.body.Load().([]byte), &got))
	think, ok := got["think"]
	return think, ok
}

func ask(t *testing.T, llm *LLM, opts ...llms.CallOption) (*llms.ContentResponse, error) {
	t.Helper()
	return llm.GenerateContent(t.Context(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, opts...)
}

func reasoningWarnings(resp *llms.ContentResponse) []llms.Warning {
	var found []llms.Warning
	for _, w := range resp.Warnings {
		if w.Option == "WithReasoning" {
			found = append(found, w)
		}
	}
	return found
}

func TestAModelTheServerListsWithoutThinkingGetsNoThink(t *testing.T) {
	t.Parallel()

	s, llm := newShowServer(t, `{"capabilities":["completion","tools","vision"]}`)
	resp, err := ask(t, llm, llms.WithReasoning(llms.ReasoningHigh, 0))
	require.NoError(t, err)

	_, sent := s.sentThink(t)
	assert.False(t, sent, "a local server answers 400 to think on a model without the thinking capability")
	assert.Equal(t, []llms.Warning{{
		Kind: llms.WarningDrop, Option: "WithReasoning", Model: "m", Asked: "high",
		Reason: "the server lists no thinking capability for this model and refuses think for it",
	}}, reasoningWarnings(resp))
}

func TestTheThinkValueFollowsTheValuesTheServerLists(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name   string
		values string
		effort llms.ReasoningEffort
		sent   any
	}{
		{"a listed level travels as itself", `[false,"high","max"],"default":"high"`, llms.ReasoningHigh, "high"},
		{"an unlisted level becomes the nearest lower one on a tie", `["low","high","max"],"default":"max"`, llms.ReasoningMedium, "low"},
		{"an effort above the list becomes its top level", `["low","medium","high"],"default":"medium"`, llms.ReasoningXHigh, "high"},
		{"a model that lists only switches gets true", `[false,true],"default":true`, llms.ReasoningHigh, true},
		{"a model that lists only true gets true", `[true],"default":true`, llms.ReasoningLow, true},
		{"an effort above every listed level takes true when true is listed", `[false,true,"medium"],"default":true`, llms.ReasoningHigh, true},
		{"an effort below the listed levels takes the nearest level beside true", `[false,true,"medium"],"default":true`, llms.ReasoningLow, "medium"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			s, llm := newShowServer(t,
				`{"capabilities":["completion","thinking","tools"],"thinking":{"values":`+tc.values+`}}`)
			resp, err := ask(t, llm, llms.WithReasoning(tc.effort, 0))
			require.NoError(t, err)

			think, sent := s.sentThink(t)
			require.True(t, sent)
			assert.Equal(t, tc.sent, think)

			if tc.sent == string(tc.effort) {
				assert.Empty(t, reasoningWarnings(resp))
				return
			}
			assert.Equal(t, []llms.Warning{{
				Kind: llms.WarningSubstitute, Option: "WithReasoning", Model: "m",
				Asked: string(tc.effort), Sent: jsonText(tc.sent),
				Reason: "the server lists the think values this model takes and this effort is not among them",
			}}, reasoningWarnings(resp))
		})
	}
}

func jsonText(v any) string {
	b, _ := json.Marshal(v)
	var s string
	if json.Unmarshal(b, &s) == nil {
		return s
	}
	return string(b)
}

func TestTurningOffAModelThatListsNoFalseIsRefusedBeforeTheChat(t *testing.T) {
	t.Parallel()

	for _, values := range []string{`["low","high","max"],"default":"max"`, `[true],"default":true`,
		`["low","medium","high"],"default":"medium"`} {
		t.Run(values, func(t *testing.T) {
			t.Parallel()

			s, llm := newShowServer(t,
				`{"capabilities":["completion","thinking"],"thinking":{"values":`+values+`}}`)
			_, err := ask(t, llm, llms.WithReasoningDisabled())

			var refused *reasoning.ErrReasoningOffUnsupported
			require.ErrorAs(t, err, &refused)
			assert.Zero(t, s.chats.Load(), "false would leave the thinking in the answer text")
		})
	}
}

func TestTurningOffAModelThatListsFalseSendsFalse(t *testing.T) {
	t.Parallel()

	s, llm := newShowServer(t,
		`{"capabilities":["completion","thinking"],"thinking":{"values":[false,"low","high"],"default":"low"}}`)
	_, err := ask(t, llm, llms.WithReasoningDisabled())
	require.NoError(t, err)

	think, sent := s.sentThink(t)
	require.True(t, sent)
	assert.Equal(t, false, think)
}

func TestAThinkingModelWithoutAUsableDescriptorKeepsTheNamedLevels(t *testing.T) {
	t.Parallel()

	for name, show := range map[string]string{
		"no descriptor":                         `{"capabilities":["completion","thinking","tools"]}`,
		"a default outside the values it lists": `{"capabilities":["thinking"],"thinking":{"values":["low","high"],"default":"max"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			s, llm := newShowServer(t, show)
			_, err := ask(t, llm, llms.WithReasoning(llms.ReasoningXHigh, 0))
			require.NoError(t, err)

			think, sent := s.sentThink(t)
			require.True(t, sent)
			assert.Equal(t, true, think, "without a usable descriptor only low, medium, high and max travel as levels")
		})
	}
}

func TestTheServerIsAskedOncePerModelAndOnlyWhenThinkingIsSet(t *testing.T) {
	t.Parallel()

	s, llm := newShowServer(t, `{"capabilities":["completion","thinking"],"thinking":{"values":[false,true],"default":true}}`)

	_, err := ask(t, llm)
	require.NoError(t, err)
	_, err = ask(t, llm, llms.WithAdaptiveReasoning(""))
	require.NoError(t, err)
	assert.Zero(t, s.shows.Load(), "a call that sets no thinking or leaves the depth to the model costs no extra request")

	for range 3 {
		_, err = ask(t, llm, llms.WithReasoning(llms.ReasoningHigh, 0))
		require.NoError(t, err)
	}
	assert.Equal(t, int32(1), s.shows.Load())
}

func TestABudgetWithoutAnEffortIsReportedOnceOnADescribedModel(t *testing.T) {
	t.Parallel()

	s, llm := newShowServer(t, `{"capabilities":["thinking"],"thinking":{"values":[false,"medium","high"],"default":"high"}}`)
	resp, err := ask(t, llm, llms.WithMaxTokens(4096), llms.WithReasoning(llms.ReasoningNone, 800))
	require.NoError(t, err)

	think, sent := s.sentThink(t)
	require.True(t, sent)
	warnings := reasoningWarnings(resp)
	require.Len(t, warnings, 1, "%v", warnings)
	assert.Equal(t, "800", warnings[0].Asked)
	assert.Equal(t, jsonText(think), warnings[0].Sent)
}

func TestABudgetOnAModelThatDoesNotThinkIsReportedAsTheBudget(t *testing.T) {
	t.Parallel()

	_, llm := newShowServer(t, `{"capabilities":["completion"]}`)
	resp, err := ask(t, llm, llms.WithMaxTokens(4096), llms.WithReasoning(llms.ReasoningNone, 800))
	require.NoError(t, err)

	warnings := reasoningWarnings(resp)
	require.Len(t, warnings, 1, "%v", warnings)
	assert.Equal(t, "800", warnings[0].Asked)
	assert.Equal(t, llms.WarningDrop, warnings[0].Kind)
}

func TestAModelWhoseOnlyValueIsFalseGetsNoThink(t *testing.T) {
	t.Parallel()

	s, llm := newShowServer(t, `{"thinking":{"values":[false],"default":false}}`)
	resp, err := ask(t, llm, llms.WithReasoning(llms.ReasoningHigh, 0))
	require.NoError(t, err)

	_, sent := s.sentThink(t)
	assert.False(t, sent, "the docs read values [false] as a model that does not think")
	require.Len(t, reasoningWarnings(resp), 1)
	assert.Equal(t, llms.WarningDrop, reasoningWarnings(resp)[0].Kind)
}

func TestADescriptorCountsWithoutACapabilityList(t *testing.T) {
	t.Parallel()

	s, llm := newShowServer(t, `{"thinking":{"values":["low","high","max"],"default":"max"}}`)
	_, err := ask(t, llm, llms.WithReasoning(llms.ReasoningMedium, 0))
	require.NoError(t, err)

	think, sent := s.sentThink(t)
	require.True(t, sent)
	assert.Equal(t, "low", think)
}

func TestTurningOffAModelWithoutTheThinkingCapabilitySendsFalse(t *testing.T) {
	t.Parallel()

	s, llm := newShowServer(t, `{"capabilities":["completion","tools"]}`)
	_, err := ask(t, llm, llms.WithReasoningDisabled())
	require.NoError(t, err)

	think, sent := s.sentThink(t)
	require.True(t, sent)
	assert.Equal(t, false, think, "the server refuses only a truthy think on such a model")
}

func TestAShowThatFailsStopsTheCallBeforeTheChat(t *testing.T) {
	t.Parallel()

	var chats atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/chat" {
			chats.Add(1)
		}
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `{"error":"model 'm' not found"}`)
	}))
	t.Cleanup(srv.Close)

	llm, err := New(WithServerURL(srv.URL), WithModel("m"))
	require.NoError(t, err)

	_, err = ask(t, llm, llms.WithReasoning(llms.ReasoningHigh, 0))
	require.ErrorContains(t, err, `ollama: show model "m"`)
	assert.Zero(t, chats.Load())
}
