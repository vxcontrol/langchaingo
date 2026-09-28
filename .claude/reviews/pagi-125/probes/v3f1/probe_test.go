package googleai

import (
	"context"
	"testing"
)

const credJSON = `{"type":"authorized_user","client_id":"cid","client_secret":"sec","refresh_token":"rt"}`

func TestProbeV3F1(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	_, err := New(context.Background(), WithAPIKey("SECRETKEY"), WithCredentialsJSON([]byte(credJSON)))
	t.Logf("key+json err: %v", err)
	t.Setenv("GOOGLE_API_KEY", "ENVSECRET")
	_, err = New(context.Background(), WithCredentialsJSON([]byte(credJSON)))
	t.Logf("envkey+json err: %v", err)
}
