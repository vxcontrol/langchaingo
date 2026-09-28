package openai_test

import (
	"encoding/json"
	"net/http"
	"net/url"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/openai"
)

type redirectRT struct{ target *url.URL }

func (r redirectRT) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.URL.Scheme = r.target.Scheme
	req.URL.Host = r.target.Host
	return http.DefaultTransport.RoundTrip(req)
}

func TestProbeOff(t *testing.T) {
	type c struct {
		name  string
		model string
		extra []openai.Option
		opts  []llms.CallOption
	}
	cases := []c{
		{"glm46-off", "z-ai/glm-4.6", []openai.Option{openai.WithModernReasoningFormat()}, []llms.CallOption{llms.WithReasoningDisabled()}},
		{"glm46-off-legacy", "z-ai/glm-4.6", nil, []llms.CallOption{llms.WithReasoningDisabled()}},
		{"kimi-k2.5-off", "moonshotai/kimi-k2.5", nil, []llms.CallOption{llms.WithReasoningDisabled()}},
		{"qwen3-off-or", "qwen/qwen3-235b-a22b", []openai.Option{openai.WithModernReasoningFormat()}, []llms.CallOption{llms.WithReasoningDisabled()}},
		{"qwen3-8b-bare-off", "qwen3-8b", nil, []llms.CallOption{llms.WithReasoningDisabled()}},
		{"gpt5-off", "gpt-5", nil, []llms.CallOption{llms.WithReasoningDisabled()}},
		{"o3-off", "o3", nil, []llms.CallOption{llms.WithReasoningDisabled()}},
	}
	for _, tc := range cases {
		p, resp, err := probeCapture2(t, tc.model, tc.extra, tc.opts...)
		b, _ := json.Marshal(p)
		_ = resp
		t.Logf("%s: err=%v body=%s", tc.name, err, b)
	}
}
