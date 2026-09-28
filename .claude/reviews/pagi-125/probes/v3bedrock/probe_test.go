package bedrock_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func v3Capture(t *testing.T, resp string, opts ...bedrock.Option) (*bedrock.LLM, *string, *int) {
	sent := new(string)
	n := new(int)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		*sent = string(b)
		*n++
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, resp)
	}))
	t.Cleanup(srv.Close)
	client := bedrockruntime.NewFromConfig(aws.Config{
		Region:      "us-east-1",
		Credentials: credentials.NewStaticCredentialsProvider("unit", "test", ""),
	}, func(o *bedrockruntime.Options) {
		o.BaseEndpoint = aws.String(srv.URL)
		o.AuthSchemePreference = []string{"sigv4"}
	})
	llm, err := bedrock.New(append([]bedrock.Option{bedrock.WithClient(client)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return llm, sent, n
}

func TestProbeV3LegacyToolChoice(t *testing.T) {
	const resp = `{"id":"x","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	for _, c := range []string{"none", "auto", "required"} {
		llm, sent, _ := v3Capture(t, resp, bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithToolChoice(c))
		t.Logf("choice=%s err=%v body=%s", c, err, *sent)
	}
}

func TestProbeV3ConverseTextPlain(t *testing.T) {
	const resp = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`
	for _, mt := range []string{"text/plain", "application/json", "image/png"} {
		llm, sent, n := v3Capture(t, resp, bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())
		_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeSystem, "be brief"),
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.BinaryPart(mt, []byte("hello world"))}},
		})
		t.Logf("mime=%s requests=%d err=%v body=%s", mt, *n, err, *sent)
	}
}
