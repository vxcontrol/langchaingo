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

func TestProbeVFADCIdentity(t *testing.T) {
	dir := t.TempDir()
	adc := filepath.Join(dir, "adc.json")
	os.WriteFile(adc, []byte(`{"type":"authorized_user","client_id":"adc-cid","client_secret":"s","refresh_token":"r","quota_project_id":"adc-quota-project"}`), 0o600)
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", adc)
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	ts := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "callers-own-token"})
	withTS := func(o *Options) { o.ClientOptions = append(o.ClientOptions, option.WithTokenSource(ts)) }
	g, err := New(context.Background(), WithCloudProject("p"), WithCloudLocation("europe-west4"), withTS)
	if err != nil {
		fmt.Println("PROBE err", err)
		return
	}
	cc := g.client.ClientConfig()
	qp, _ := cc.Credentials.QuotaProjectID(context.Background())
	fmt.Printf("PROBE backend=%v apikey=%q creds!=nil:%v quotaProject=%q\n", cc.Backend, cc.APIKey, cc.Credentials != nil, qp)
	g2, err := New(context.Background(), WithCloudProject("p"), WithCloudLocation("europe-west4"), WithAPIKey("vertex-key"))
	if err != nil {
		fmt.Println("PROBE err2", err)
		return
	}
	cc2 := g2.client.ClientConfig()
	qp2 := ""
	if cc2.Credentials != nil {
		qp2, _ = cc2.Credentials.QuotaProjectID(context.Background())
	}
	fmt.Printf("PROBE apikey path: apikey=%q quotaProject=%q\n", cc2.APIKey, qp2)
}
