package bedrock_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestAStrayBearerTokenDoesNotResignTheReplay(t *testing.T) {
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "someone-elses-token")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, legacyAnswer)
	}))
	t.Cleanup(srv.Close)

	tap := &authHeaderTap{next: http.DefaultTransport}
	cfg, err := config.LoadDefaultConfig(t.Context(),
		append([]func(*config.LoadOptions) error{
			config.WithHTTPClient(&http.Client{Transport: tap}),
		}, replayCredentials(false)...)...)
	require.NoError(t, err)

	client := bedrockruntime.NewFromConfig(cfg,
		append(replayClientOptions(false), func(o *bedrockruntime.Options) {
			o.BaseEndpoint = aws.String(srv.URL)
		})...)

	llm, err := bedrock.New(bedrock.WithClient(client),
		bedrock.WithModel("anthropic.claude-sonnet-4-5-20250929-v1:0"))
	require.NoError(t, err)

	_, err = llm.GenerateContent(context.Background(),
		[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "how many rooms are free?")})
	require.NoError(t, err)

	assert.Equal(t, "AWS4-HMAC-SHA256", strings.SplitN(tap.authorization, " ", 2)[0],
		"a bearer token in the environment must not change how a replayed request is signed")
}
