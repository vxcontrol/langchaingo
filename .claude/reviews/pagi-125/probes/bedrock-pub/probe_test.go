package bedrock_test

import (
	"context"
	"encoding/json"
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

func probeLLM(t *testing.T, respBody string, opts ...bedrock.Option) (*bedrock.LLM, *[]string) {
	t.Helper()
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, r.URL.Path+" "+string(b))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, respBody)
	}))
	t.Cleanup(srv.Close)
	client := bedrockruntime.NewFromConfig(aws.Config{
		Region:      "us-east-1",
		Credentials: credentials.NewStaticCredentialsProvider("unit", "test", ""),
	}, func(o *bedrockruntime.Options) { o.BaseEndpoint = aws.String(srv.URL) })
	llm, err := bedrock.New(append([]bedrock.Option{bedrock.WithClient(client)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return llm, &bodies
}

const converseOK = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`

func toolConfigOf(body string) string {
	var p map[string]any
	idx := 0
	for i, c := range body {
		if c == ' ' {
			idx = i + 1
			break
		}
	}
	_ = json.Unmarshal([]byte(body[idx:]), &p)
	b, _ := json.Marshal(p["toolConfig"].(map[string]any)["toolChoice"])
	return string(b)
}

func TestProbeToolChoiceConverse(t *testing.T) {
	tool := llms.Tool{Type: "function", Function: &llms.FunctionDefinition{
		Name: "get_weather", Description: "w",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
	}}
	for _, model := range []string{"us.meta.llama4-maverick-17b-instruct-v1:0", "openai.gpt-oss-120b-1:0", "deepseek.v3.2"} {
		for _, choice := range []any{"required", llms.ToolChoice{Type: "function", Function: &llms.FunctionReference{Name: "get_weather"}}} {
			llm, bodies := probeLLM(t, converseOK, bedrock.WithModel(model), bedrock.WithConverseAPI())
			_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "weather in Paris?")},
				llms.WithTools([]llms.Tool{tool}), llms.WithToolChoice(choice))
			tc := ""
			if len(*bodies) > 0 {
				tc = toolConfigOf((*bodies)[0])
			}
			t.Logf("model=%s choice=%v err=%v toolChoiceOnWire=%s", model, choice, err, tc)
		}
	}
}

const novaLegacyOK = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1}}`

func TestProbeNovaDelegation(t *testing.T) {
	for _, converse := range []bool{false, true} {
		opts := []bedrock.Option{bedrock.WithModel("us.amazon.nova-2-lite-v1:0")}
		resp := novaLegacyOK
		if converse {
			opts = append(opts, bedrock.WithConverseAPI())
			resp = converseOK
		}
		llm, bodies := probeLLM(t, resp, opts...)
		_, err := llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")},
			llms.WithAdaptiveReasoning(llms.ReasoningNone), llms.WithMaxTokens(500), llms.WithTemperature(0.3))
		b := ""
		if len(*bodies) > 0 {
			b = (*bodies)[0]
		}
		t.Logf("converse=%v err=%v body=%s", converse, err, b)
	}
}

func TestProbeBinaryTextConverse(t *testing.T) {
	for _, mime := range []string{"text/plain", "application/json", "text/markdown"} {
		llm, bodies := probeLLM(t, converseOK, bedrock.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0"), bedrock.WithConverseAPI())
		_, err := llm.GenerateContent(context.Background(), []llms.MessageContent{{
			Role:  llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{llms.TextPart("summarize the attached file"), llms.BinaryPart(mime, []byte("quarterly revenue grew 12%"))},
		}})
		b := ""
		if len(*bodies) > 0 {
			b = (*bodies)[0]
		}
		t.Logf("mime=%s err=%v body=%s", mime, err, b)
	}
}
