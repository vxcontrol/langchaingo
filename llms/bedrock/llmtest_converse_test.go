package bedrock_test

import (
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/smithy-go/auth/bearer"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/testing/llmtest"
)

const awsBedrockHost = "bedrock-runtime.us-east-1.amazonaws.com"

func conformanceClient(t *testing.T, rr *httprr.RecordReplay) *bedrockruntime.Client {
	t.Helper()

	base := os.Getenv("BEDROCK_PASSTHROUGH_URL")
	prefix := ""
	if base != "" {
		parsed, err := url.Parse(base)
		require.NoError(t, err, "BEDROCK_PASSTHROUGH_URL must be a URL")
		prefix = strings.TrimSuffix(parsed.Path, "/")
	}

	rr.ScrubReq(func(req *http.Request) error {
		req.Header.Del("Amz-Sdk-Invocation-Id")
		req.Header.Del("Amz-Sdk-Request")
		req.Header.Del("X-Amz-Date")
		req.Header.Del("Authorization")
		req.URL.Scheme = "https"
		req.URL.Host = awsBedrockHost
		req.URL.Path = strings.TrimPrefix(req.URL.Path, prefix)
		req.URL.RawPath = ""
		req.Host = awsBedrockHost
		return nil
	})

	httpClient := &http.Client{Transport: rr}

	if !rr.Recording() {
		cfg, err := config.LoadDefaultConfig(t.Context(),
			append([]func(*config.LoadOptions) error{config.WithHTTPClient(httpClient)},
				replayCredentials(false)...)...)
		require.NoError(t, err)
		return bedrockruntime.NewFromConfig(cfg, replayClientOptions(false)...)
	}

	key := os.Getenv("BEDROCK_PASSTHROUGH_KEY")
	require.NotEmpty(t, base, "recording needs BEDROCK_PASSTHROUGH_URL")
	require.NotEmpty(t, key, "recording needs BEDROCK_PASSTHROUGH_KEY")

	cfg, err := config.LoadDefaultConfig(t.Context(),
		config.WithHTTPClient(httpClient), config.WithRegion(replayRegion))
	require.NoError(t, err)

	return bedrockruntime.NewFromConfig(cfg, func(o *bedrockruntime.Options) {
		o.BaseEndpoint = aws.String(base)
		o.BearerAuthTokenProvider = bearer.StaticTokenProvider{Token: bearer.Token{Value: key}}
		o.AuthSchemePreference = []string{"httpBearerAuth"}
	})
}

func TestLLMConverse(t *testing.T) {
	cassette := filepath.Join("testdata", "TestLLMConverse.httprr")
	recording, err := httprr.Recording(cassette)
	if err != nil {
		t.Fatal(err)
	}
	if _, statErr := os.Stat(cassette); statErr != nil && !recording {
		t.Skip("no httprr recording for TestLLMConverse; re-run with -httprecord=. and a gateway key")
	}

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	// llmtest drives parallel subtests, which resume only after this function
	// returns; a deferred Close would shut the recorder before they run.
	t.Cleanup(func() {
		if closeErr := rr.Close(); closeErr != nil {
			t.Errorf("closing the recording: %v", closeErr)
		}
	})

	llm, err := bedrock.New(bedrock.WithClient(conformanceClient(t, rr)),
		bedrock.WithModel(bedrock.ModelAnthropicClaudeHaiku45), bedrock.WithConverseAPI())
	if err != nil {
		t.Fatalf("Failed to create Bedrock LLM: %v", err)
	}

	llmtest.TestLLM(t, llm)
}

func TestTheConformanceRecordingAddressesTheVendor(t *testing.T) {
	t.Parallel()

	recorded, err := os.ReadFile(filepath.Join("testdata", "TestLLMConverse.httprr"))
	if err != nil {
		t.Skip("no recording for TestLLMConverse yet")
	}

	requests := 0
	for _, line := range strings.Split(string(recorded), "\n") {
		switch {
		case strings.HasPrefix(line, "POST "):
			requests++
			require.Contains(t, line, awsBedrockHost,
				"a recording made through a gateway must be stored under the vendor's own address")
		case strings.HasPrefix(line, "Authorization:"):
			require.Fail(t, "the recording carries a credential header")
		}
	}
	require.NotZero(t, requests, "the recording must hold at least one request")
}
