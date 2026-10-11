package anthropic

import (
	"cmp"
	"context"
	"net/http"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/httputil"
	"github.com/vxcontrol/langchaingo/internal/httprr"
)

func TestTheReportIsReadFromTheVendorsModelAnswer(t *testing.T) {
	t.Parallel()

	httprr.SkipIfNoCredentialsAndRecordingMissing(t, tokenEnvVarName)
	var transport http.RoundTripper = httputil.DefaultTransport
	if base := os.Getenv("ANTHROPIC_BASE_URL"); base != "" {
		transport = &httputil.ApiKeyTransport{Transport: transport, BaseURL: base}
	}
	rr := httprr.OpenForTest(t, transport)
	t.Cleanup(func() { rr.Close() })

	token := "test-api-key"
	if !rr.Replaying() {
		token = cmp.Or(os.Getenv(tokenEnvVarName), token)
	}
	llm, err := New(WithToken(token), WithHTTPClient(rr.Client()))
	require.NoError(t, err)

	info, err := llm.client.GetModel(context.Background(), "claude-sonnet-5-5")
	require.NoError(t, err)
	report := claudeReportFrom(*info)

	assert.True(t, report.Reported)
	assert.True(t, report.Thinking)
	assert.True(t, report.Adaptive)
	assert.False(t, report.Budget)
	if assert.NotNil(t, report.Disable, "the answer says whether thinking can be disabled") {
		assert.False(t, *report.Disable)
	}
	assert.True(t, report.Effort)
	assert.Equal(t, []string{"low", "medium", "high", "xhigh", "max"}, report.Efforts)
	assert.True(t, report.StructuredOutputs)
}
