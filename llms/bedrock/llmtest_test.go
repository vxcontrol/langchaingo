package bedrock_test

import (
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/vxcontrol/langchaingo/internal/httprr"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
	"github.com/vxcontrol/langchaingo/testing/llmtest"
)

func TestLLM(t *testing.T) {
	cassette := filepath.Join("testdata", "TestLLM.httprr")
	recording, err := httprr.Recording(cassette)
	if err != nil {
		t.Fatal(err)
	}
	if _, statErr := os.Stat(cassette); statErr != nil && !recording {
		t.Skip("no httprr recording for TestLLM; re-run with -httprecord=. and a gateway key")
	}

	rr := httprr.OpenForTest(t, http.DefaultTransport)
	// llmtest drives parallel subtests, which resume only after this function
	// returns; a deferred Close would shut the recorder before they run.
	t.Cleanup(func() {
		if closeErr := rr.Close(); closeErr != nil {
			t.Errorf("closing the recording: %v", closeErr)
		}
	})

	llm, err := bedrock.New(bedrock.WithClient(conformanceClient(t, rr)),
		bedrock.WithModel(bedrock.ModelAnthropicClaudeHaiku45))
	if err != nil {
		t.Fatalf("Failed to create Bedrock LLM: %v", err)
	}

	llmtest.TestLLM(t, llm)
}
