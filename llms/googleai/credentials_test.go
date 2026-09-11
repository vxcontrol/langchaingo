package googleai

import (
	"os"
	"path/filepath"
	"testing"

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
