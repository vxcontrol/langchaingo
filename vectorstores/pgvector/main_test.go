package pgvector

import (
	"context"
	"os"
	"testing"

	"github.com/vxcontrol/langchaingo/internal/testutil/testctr"
)

func TestMain(m *testing.M) {
	code := testctr.EnsureTestEnv()
	if code == 0 {
		code = m.Run()
	}
	if sharedPostgres.container != nil {
		_ = sharedPostgres.container.Terminate(context.Background())
	}
	os.Exit(code)
}
