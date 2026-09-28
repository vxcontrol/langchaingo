package googleai

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/oauth2"
	"google.golang.org/api/option"
)

func TestProbeCustomClientOptionIgnored(t *testing.T) {
	dir := t.TempDir()
	adc := filepath.Join(dir, "adc.json")
	_ = os.WriteFile(adc, []byte(`{"type":"authorized_user","client_id":"adc-client.apps.googleusercontent.com","client_secret":"s","refresh_token":"r","quota_project_id":"adc-quota-project"}`), 0o600)
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", adc)
	t.Setenv("GOOGLE_API_KEY", "")

	callerTS := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "callers-own-token"})
	withTS := func(o *Options) { o.ClientOptions = append(o.ClientOptions, option.WithTokenSource(callerTS)) }

	g, err := New(context.Background(), WithCloudProject("p"), WithCloudLocation("europe-west4"), withTS)
	fmt.Printf("TOKENSOURCE: err=%v\n", err)
	if err == nil {
		cfg := g.client.ClientConfig()
		qp, _ := cfg.Credentials.QuotaProjectID(context.Background())
		fmt.Printf("TOKENSOURCE: credentials used come from ADC file (quota project %q), caller token source ignored\n", qp)
	}

	g2, err := New(context.Background(), WithCloudProject("p"), WithCloudLocation("europe-west4"), WithAPIKey("vertex-key"))
	fmt.Printf("APIKEY on vertex: err=%v\n", err)
	if err == nil {
		cfg := g2.client.ClientConfig()
		fmt.Printf("APIKEY on vertex: config.APIKey=%q credentials nil=%v\n", cfg.APIKey, cfg.Credentials == nil)
	}
}
