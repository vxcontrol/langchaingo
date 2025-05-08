package documentloaders

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTranscription(t *testing.T) {
	t.Parallel()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	t.Run("Test with local file", func(t *testing.T) {
		t.Parallel()
		audioFilePath := "./testdata/sample.mp3"
		_, err := os.Stat(audioFilePath)
		require.NoError(t, err)
		opts := []WhisperOpenAIOption{
			WithAudioPath(audioFilePath),
		}
		whisper := NewWhisperOpenAI(os.Getenv("OPENAI_API_KEY"), opts...)

		rsp, err := whisper.Load(t.Context())
		require.NoError(t, err)

		assert.NotEmpty(t, rsp)
	})

	t.Run("Test from url", func(t *testing.T) {
		t.Parallel()

		audioURL := "https://raw.githubusercontent.com/vxcontrol/langchaingo/main-vxcontrol/documentloaders/testdata/sample.mp3"

		opts := []WhisperOpenAIOption{
			WithAudioPath(audioURL),
		}
		whisper := NewWhisperOpenAI(os.Getenv("OPENAI_API_KEY"), opts...)

		rsp, err := whisper.Load(t.Context())
		require.NoError(t, err)

		assert.NotEmpty(t, rsp)
	})
}
