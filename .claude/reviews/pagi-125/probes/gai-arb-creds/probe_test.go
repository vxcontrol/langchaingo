package googleai

import (
	"context"
	"testing"
)

const probeCreds = `{"type":"authorized_user","client_id":"id.apps.googleusercontent.com","client_secret":"secret","refresh_token":"token"}`

func TestProbeArbExplicitKey(t *testing.T) {
	_, err := New(context.Background(), WithAPIKey("k"), WithCredentialsJSON([]byte(probeCreds)))
	t.Logf("explicit key + creds: err=%v", err)
}

func TestProbeArbEnvKey(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "envkey")
	_, err := New(context.Background(), WithCredentialsJSON([]byte(probeCreds)))
	t.Logf("env key + creds: err=%v", err)
}

func TestProbeArbCredsOnly(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	_, err := New(context.Background(), WithCredentialsJSON([]byte(probeCreds)))
	t.Logf("creds only: err=%v", err)
}
