package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeToolMsgTextPart(t *testing.T) {
	for name, parts := range map[string][]llms.ContentPart{
		"resp+text": {llms.ToolCallResponse{ToolCallID: "a", Name: "lookup", Content: "1"}, llms.TextContent{Text: "note"}},
		"text+resp": {llms.TextContent{Text: "note"}, llms.ToolCallResponse{ToolCallID: "a", Name: "lookup", Content: "1"}},
	} {
		m, err := handleToolMessage(llms.MessageContent{Role: llms.ChatMessageTypeTool, Parts: parts})
		b, _ := json.Marshal(m)
		t.Logf("%s: err=%v msg=%s", name, err, b)
	}
}
