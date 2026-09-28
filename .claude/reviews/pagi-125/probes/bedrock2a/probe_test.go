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

func probeLLM(t *testing.T, body string, opts ...bedrock.Option) (*bedrock.LLM, *string) {
	t.Helper()
	sent := new(string)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		*sent = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
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
	return llm, sent
}

const probeLegacyAnswer = `{"id":"x","type":"message","role":"assistant","model":"m",` +
	`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",` +
	`"usage":{"input_tokens":1,"output_tokens":1}}`

const probeConverseAnswer = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},` +
	`"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`

func TestProbeLegacyToolChoiceWithoutTools(t *testing.T) {
	for _, choice := range []any{"none", "auto", "required"} {
		llm, sent := probeLLM(t, probeLegacyAnswer, bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"))
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithToolChoice(choice))
		t.Logf("choice=%v err=%v\nbody=%s", choice, err, *sent)
	}
}

func TestProbeConverseBinaryTextAlone(t *testing.T) {
	llm, sent := probeLLM(t, probeConverseAnswer, bedrock.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeSystem, "Summarize the document the user sends."),
			{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{
				llms.BinaryPart("text/plain", []byte("hello world")),
			}},
		})
	t.Logf("err=%v resp_nil=%v\nbody=%s", err, resp == nil, *sent)
}

func TestProbeConverseBinaryText(t *testing.T) {
	llm, sent := probeLLM(t, probeConverseAnswer, bedrock.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())
	resp, err := llm.GenerateContent(context.Background(),
		[]llms.MessageContent{{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{
			llms.TextPart("Summarize this file:"),
			llms.BinaryPart("text/plain", []byte("hello world")),
		}}})
	t.Logf("err=%v resp_nil=%v\nbody=%s", err, resp == nil, *sent)
}
