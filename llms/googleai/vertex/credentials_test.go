package vertex_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vxcontrol/langchaingo/llms/googleai"
	"github.com/vxcontrol/langchaingo/llms/googleai/vertex"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const refreshTokenCredentials = `{"type":"authorized_user","client_id":"id.apps.googleusercontent.com",` +
	`"client_secret":"secret","refresh_token":"token"}`

func TestVertexTakesTheServiceAccountTheCallerNames(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "credentials.json")
	require.NoError(t, os.WriteFile(path, []byte(refreshTokenCredentials), 0o600))

	for name, opt := range map[string]googleai.Option{
		"WithCredentialsFile": googleai.WithCredentialsFile(path),
		"WithCredentialsJSON": googleai.WithCredentialsJSON([]byte(refreshTokenCredentials)),
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			client, err := vertex.New(t.Context(),
				googleai.WithCloudProject("p"), googleai.WithCloudLocation("europe-west4"), opt)
			require.NoError(t, err)
			require.NotNil(t, client)
		})
	}
}

func TestVertexStillRefusesAGRPCConnectionAndSaysWhatToUse(t *testing.T) {
	t.Parallel()

	_, err := vertex.New(t.Context(),
		googleai.WithCloudProject("p"), googleai.WithCloudLocation("europe-west4"),
		googleai.WithGRPCConn(nil))

	var notHonored *googleai.ErrOptionNotHonored
	require.ErrorAs(t, err, &notHonored)
	assert.Contains(t, err.Error(), "WithHTTPClient",
		"the error must name the replacement, not only the refusal")
}
