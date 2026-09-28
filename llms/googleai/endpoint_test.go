package googleai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

func TestAnEndpointInHostPortFormStillReachesTheServer(t *testing.T) {
	t.Parallel()

	var gotPath string
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{}}`)
	}))
	t.Cleanup(server.Close)

	hostPort := strings.TrimPrefix(server.URL, "https://")
	llm, err := New(t.Context(), WithAPIKey("k"), WithEndpoint(hostPort),
		WithHTTPClient(server.Client()), WithDefaultModel("gemini-2.5-flash"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(t.Context(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	require.NoError(t, err)
	assert.Contains(t, gotPath, "/models/gemini-2.5-flash:generateContent")
}
