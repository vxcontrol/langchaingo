package bedrock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func vbrLLM(t *testing.T, opts ...bedrock.Option) (*bedrock.LLM, *[]string) {
	t.Helper()
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, r.URL.Path+" "+string(b))
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/converse") {
			_, _ = io.WriteString(w, `{"output":{"message":{"role":"assistant","content":[{"text":"{\"answer\":\"ok\"}"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
			return
		}
		_, _ = io.WriteString(w, `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1}}`)
	}))
	t.Cleanup(srv.Close)
	client := bedrockruntime.NewFromConfig(aws.Config{Region: "us-east-1", Credentials: credentials.NewStaticCredentialsProvider("a", "b", "")},
		func(o *bedrockruntime.Options) { o.BaseEndpoint = aws.String(srv.URL) })
	llm, err := bedrock.New(append([]bedrock.Option{bedrock.WithClient(client)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return llm, &bodies
}

func vbrMsgs() []llms.MessageContent {
	return []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")}
}

func vbrField(body string, keys ...string) string {
	i := strings.Index(body, " ")
	var p any
	_ = json.Unmarshal([]byte(body[i+1:]), &p)
	for _, k := range keys {
		m, ok := p.(map[string]any)
		if !ok {
			return "<none>"
		}
		p = m[k]
	}
	b, _ := json.Marshal(p)
	return string(b)
}

func TestProbeVbrF1(t *testing.T) {
	schema := `{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`
	for _, m := range []string{
		"arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4e5f6",
		"arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abcdef123456",
		"us.anthropic.claude-sonnet-4-5-20250929-v1:0",
	} {
		llm, bodies := vbrLLM(t, bedrock.WithConverseAPI(), bedrock.WithModel(m), bedrock.WithModelProvider("anthropic"))
		_, err := llm.GenerateContent(context.Background(), vbrMsgs(),
			llms.WithStructuredOutput(llms.StructuredOutputConfig{Name: "a", Schema: json.RawMessage(schema)}))
		oc := "<not sent>"
		if len(*bodies) > 0 {
			oc = vbrField((*bodies)[0], "outputConfig")
		}
		t.Logf("F1 model=%s err=%v requests=%d outputConfig=%s", m, err, len(*bodies), oc)
	}
}

func TestProbeVbrF2(t *testing.T) {
	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{Name: "get_weather", Description: "w",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}}}}
	for _, m := range []string{"us.meta.llama4-maverick-17b-instruct-v1:0", "openai.gpt-oss-120b-1:0", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"} {
		for _, ch := range []any{"required", llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "get_weather"}}} {
			llm, bodies := vbrLLM(t, bedrock.WithConverseAPI(), bedrock.WithModel(m))
			_, err := llm.GenerateContent(context.Background(), vbrMsgs(), llms.WithTools([]llms.Tool{tool}), llms.WithToolChoice(ch))
			tc := "<not sent>"
			if len(*bodies) > 0 {
				tc = vbrField((*bodies)[0], "toolConfig", "toolChoice")
			}
			t.Logf("F2 model=%s choice=%v err=%v toolChoice=%s", m, ch, err, tc)
		}
	}
}

func TestProbeVbrF3(t *testing.T) {
	for _, conv := range []bool{false, true} {
		opts := []bedrock.Option{bedrock.WithModel("us.amazon.nova-2-lite-v1:0")}
		if conv {
			opts = append(opts, bedrock.WithConverseAPI())
		}
		llm, bodies := vbrLLM(t, opts...)
		_, err := llm.GenerateContent(context.Background(), vbrMsgs(),
			llms.WithAdaptiveReasoning(llms.ReasoningNone), llms.WithMaxTokens(500), llms.WithTemperature(0.3), llms.WithTopP(0.9))
		b := "<not sent>"
		if len(*bodies) > 0 {
			b = (*bodies)[0]
		}
		t.Logf("F3 converse=%v err=%v body=%s", conv, err, b)
	}
}
