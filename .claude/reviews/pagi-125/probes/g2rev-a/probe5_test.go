package googleai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestProbeImageOff(t *testing.T) {
	for _, model := range []string{"gemini-2.5-flash-image", "gemini-3.1-flash-image-preview", "gemini-2.5-flash-preview-tts"} {
		rt := &probe4RT{}
		llm, err := New(context.Background(), WithAPIKey("k"), WithDefaultModel(model),
			WithHTTPClient(&http.Client{Transport: rt}))
		if err != nil {
			t.Fatal(err)
		}
		_, err = llm.GenerateContent(context.Background(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "draw a cat")},
			llms.WithReasoningDisabled())
		var m map[string]any
		_ = json.Unmarshal(rt.sent, &m)
		gc, _ := m["generationConfig"].(map[string]any)
		b, _ := json.Marshal(gc["thinkingConfig"])
		fmt.Printf("OFF %s: err=%v thinkingConfig=%s\n", model, err, b)
	}
}
