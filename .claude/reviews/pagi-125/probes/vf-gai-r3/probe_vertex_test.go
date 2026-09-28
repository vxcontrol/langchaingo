package vertex_test

import (
	"context"
	"fmt"
	"testing"

	"golang.org/x/oauth2"
	"google.golang.org/api/option"

	"github.com/vxcontrol/langchaingo/llms/googleai"
	"github.com/vxcontrol/langchaingo/llms/googleai/vertex"
)

func TestProbeVFTokenSource(t *testing.T) {
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/adc.json")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	ts := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "callers-own-token"})
	withTS := func(o *googleai.Options) { o.ClientOptions = append(o.ClientOptions, option.WithTokenSource(ts)) }
	v, err := vertex.New(context.Background(), googleai.WithCloudProject("p"), googleai.WithCloudLocation("europe-west4"), withTS)
	fmt.Printf("PROBE tokensource: client=%v err=%v\n", v != nil, err)
	v2, err2 := vertex.New(context.Background(), googleai.WithCloudProject("p"), googleai.WithCloudLocation("europe-west4"), googleai.WithAPIKey("vertex-key"))
	fmt.Printf("PROBE apikey: client=%v err=%v\n", v2 != nil, err2)
}
