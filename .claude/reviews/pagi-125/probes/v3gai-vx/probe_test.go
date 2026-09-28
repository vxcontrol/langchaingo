package vertex_test

import (
	"context"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/googleai"
	"github.com/vxcontrol/langchaingo/llms/googleai/vertex"
)

const cj = `{"type":"authorized_user","client_id":"id.apps.googleusercontent.com","client_secret":"secret","refresh_token":"token"}`

func TestProbeVertexEndpoint(t *testing.T) {
	c, err := vertex.New(context.Background(), googleai.WithCloudProject("p"), googleai.WithCloudLocation("us-central1"),
		googleai.WithCredentialsJSON([]byte(cj)), googleai.WithEndpoint("us-central1-aiplatform.googleapis.com:443"))
	if err != nil {
		t.Logf("new err=%v", err)
		return
	}
	_, err = c.GenerateContent(context.Background(), []llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
	t.Logf("vertex host:port endpoint: err=%v", err)
}
