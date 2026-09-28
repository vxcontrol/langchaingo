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

func TestProbeVertexTokenSource(t *testing.T) {
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/adc.json")
	callerTS := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "callers-own-token"})
	withTS := func(o *googleai.Options) { o.ClientOptions = append(o.ClientOptions, option.WithTokenSource(callerTS)) }
	v, err := vertex.New(context.Background(),
		googleai.WithCloudProject("p"), googleai.WithCloudLocation("europe-west4"), withTS)
	fmt.Printf("VERTEX+TokenSource (ADC broken): client=%v err=%v\n", v != nil, err)
}
