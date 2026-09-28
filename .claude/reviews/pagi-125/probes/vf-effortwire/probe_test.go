package openai_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

type vfDoer struct{ body string }

func (d *vfDoer) Do(r *http.Request) (*http.Response, error) {
	b, _ := io.ReadAll(r.Body)
	d.body = string(b)
	resp := `{"id":"x","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(bytes.NewBufferString(resp)), Request: r}, nil
}

type vfCase struct {
	name  string
	opts  []openai.Option
	copts []llms.CallOption
}

func vfRun(t *testing.T, c vfCase) {
	d := &vfDoer{}
	o := append([]openai.Option{openai.WithToken("k"), openai.WithHTTPClient(d)}, c.opts...)
	llm, err := openai.New(o...)
	if err != nil {
		fmt.Printf("%s: NEW ERR %v\n", c.name, err)
		return
	}
	_, err = llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}, c.copts...)
	fmt.Printf("%s:\n  err=%v\n  body=%s\n", c.name, err, d.body)
}

func TestProbeVF(t *testing.T) {
	or := "https://openrouter.ai/api/v1"
	var cases []vfCase
	// Finding 1
	for _, m := range []string{"z-ai/glm-4.6", "moonshotai/kimi-k2.5", "minimax/minimax-m2", "zai-org/GLM-4.5-Air"} {
		cases = append(cases, vfCase{"F1 OR modern budget4000 " + m, []openai.Option{openai.WithBaseURL(or), openai.WithModernReasoningFormat(), openai.WithModel(m)}, []llms.CallOption{llms.WithReasoning(llms.ReasoningNone, 4000)}})
		cases = append(cases, vfCase{"F1 OR modern high " + m, []openai.Option{openai.WithBaseURL(or), openai.WithModernReasoningFormat(), openai.WithModel(m)}, []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}})
	}
	cases = append(cases, vfCase{"F1 vLLM budget GLM-4.5-Air", []openai.Option{openai.WithBaseURL("http://localhost:8000/v1"), openai.WithModel("zai-org/GLM-4.5-Air")}, []llms.CallOption{llms.WithReasoning(llms.ReasoningNone, 4000)}})
	// Finding 2
	cases = append(cases, vfCase{"F2 azure prod-reasoner", []openai.Option{openai.WithAPIType(openai.APITypeAzure), openai.WithBaseURL("https://myres.openai.azure.com"), openai.WithModel("prod-reasoner"), openai.WithEmbeddingModel("emb")}, []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithTemperature(0.2), llms.WithTopP(0.9)}})
	for _, m := range []string{"my-gpt5-deployment", "gpt-5.3-codex", "gpt-5-codex", "gpt-5", "o3"} {
		cases = append(cases, vfCase{"F2 openai " + m, []openai.Option{openai.WithModel(m)}, []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0), llms.WithTemperature(0.2), llms.WithTopP(0.9)}})
	}
	// Finding 3
	for _, m := range []string{"qwen/qwen3-235b-a22b", "qwen/qwen3-max", "z-ai/glm-4.6", "moonshotai/kimi-k2-thinking", "minimax/minimax-m2", "x-ai/grok-code-fast-1"} {
		cases = append(cases, vfCase{"F3 OR modern high " + m, []openai.Option{openai.WithBaseURL(or), openai.WithModernReasoningFormat(), openai.WithModel(m)}, []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}})
	}
	cases = append(cases, vfCase{"F3 OR modern off qwen/qwen3-235b-a22b", []openai.Option{openai.WithBaseURL(or), openai.WithModernReasoningFormat(), openai.WithModel("qwen/qwen3-235b-a22b")}, []llms.CallOption{llms.WithReasoningDisabled()}})
	cases = append(cases, vfCase{"F3 groq off qwen/qwen3-32b", []openai.Option{openai.WithBaseURL("https://api.groq.com/openai/v1"), openai.WithModel("qwen/qwen3-32b")}, []llms.CallOption{llms.WithReasoningDisabled()}})
	cases = append(cases, vfCase{"F3 groq high qwen/qwen3-32b", []openai.Option{openai.WithBaseURL("https://api.groq.com/openai/v1"), openai.WithModel("qwen/qwen3-32b")}, []llms.CallOption{llms.WithReasoning(llms.ReasoningHigh, 0)}})
	for _, c := range cases {
		vfRun(t, c)
	}
}
