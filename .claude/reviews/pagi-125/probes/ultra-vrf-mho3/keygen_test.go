package ollama

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/pem"
	"os"
	"testing"

	"golang.org/x/crypto/ssh"
)

func TestProbeKeygen(t *testing.T) {
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	blk, err := ssh.MarshalPrivateKey(priv, "")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(os.Getenv("PROBE_KEY"), pem.EncodeToMemory(blk), 0o600); err != nil {
		t.Fatal(err)
	}
}
