package wikipedia

import (
	"testing"

	"github.com/stretchr/testify/require"
)

const _userAgent = "langchaingo test (https://github.com/vxcontrol/langchaingo)"

func TestWikipedia(t *testing.T) {
	t.Parallel()

	tool := New(_userAgent)
	_, err := tool.Call(t.Context(), "america")
	require.NoError(t, err)
}
