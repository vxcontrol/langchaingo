package googleai

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vxcontrol/langchaingo/llms"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const refreshTokenCredentials = `{"type":"authorized_user","client_id":"id.apps.googleusercontent.com",` +
	`"client_secret":"secret","refresh_token":"token"}`

func credentialsFile(t *testing.T) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "credentials.json")
	require.NoError(t, os.WriteFile(path, []byte(refreshTokenCredentials), 0o600))

	return path
}

func TestTheCallerCanNameACredentialsFile(t *testing.T) {
	t.Parallel()

	client, err := New(t.Context(),
		WithCloudProject("p"), WithCloudLocation("europe-west4"),
		WithCredentialsFile(credentialsFile(t)))
	require.NoError(t, err, "a service account is how the Vertex backend is authenticated")
	require.NotNil(t, client)
}

func TestTheCallerCanHandOverCredentialsJSON(t *testing.T) {
	t.Parallel()

	client, err := New(t.Context(),
		WithCloudProject("p"), WithCloudLocation("europe-west4"),
		WithCredentialsJSON([]byte(refreshTokenCredentials)))
	require.NoError(t, err)
	require.NotNil(t, client)
}

func TestCredentialsThatCannotBeReadAreReportedToTheCaller(t *testing.T) {
	t.Parallel()

	_, err := New(t.Context(),
		WithCloudProject("p"), WithCloudLocation("europe-west4"),
		WithCredentialsFile(filepath.Join(t.TempDir(), "missing.json")))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot be read")
}

func TestACallerWhoNamedNoCredentialsKeepsTheDefaultChain(t *testing.T) {
	t.Parallel()

	options := DefaultOptions()
	detected, err := options.detectCredentials()
	require.NoError(t, err)
	assert.Nil(t, detected, "application default credentials stay in charge")
}

func TestAnAPIKeyAuthenticatesTheGeminiAPIWhenCredentialsAreAlsoNamed(t *testing.T) {
	t.Parallel()

	var gotKey string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("x-goog-api-key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},`+
			`"finishReason":"STOP"}],"usageMetadata":{}}`)
	}))
	t.Cleanup(server.Close)

	for name, opt := range map[string]Option{
		"WithCredentialsFile": WithCredentialsFile(credentialsFile(t)),
		"WithCredentialsJSON": WithCredentialsJSON([]byte(refreshTokenCredentials)),
	} {
		llm, err := New(t.Context(), WithAPIKey("the-callers-key"), opt,
			WithEndpoint(server.URL), WithDefaultModel("gemini-2.5-flash"))
		require.NoError(t, err, name)
		assert.NotContains(t, fmt.Sprint(err), "the-callers-key", name)

		_, err = llm.GenerateContent(t.Context(),
			[]llms.MessageContent{llms.TextParts(llms.ChatMessageTypeHuman, "hi")})
		require.NoError(t, err, name)
		assert.Equal(t, "the-callers-key", gotKey, name)
	}
}
