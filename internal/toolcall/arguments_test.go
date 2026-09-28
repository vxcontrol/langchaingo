package toolcall_test

import (
	"testing"

	"github.com/vxcontrol/langchaingo/internal/toolcall"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDecodeKeepsAnIntegerWiderThanFloat64(t *testing.T) {
	t.Parallel()

	args, err := toolcall.Decode(`{"id":9007199254740993,"nested":{"amount":12345678901234567}}`)
	require.NoError(t, err)
	assert.Equal(t, int64(9007199254740993), args["id"])
	assert.Equal(t, int64(12345678901234567), args["nested"].(map[string]any)["amount"])
}

func TestDecodeFieldsKeepsTheOrderTheModelSent(t *testing.T) {
	t.Parallel()

	fields, err := toolcall.DecodeFields(`{"zebra":1,"alpha":2,"middle":3}`)
	require.NoError(t, err)
	require.Len(t, fields, 3)
	assert.Equal(t, []string{"zebra", "alpha", "middle"},
		[]string{fields[0].Key, fields[1].Key, fields[2].Key})
}

func TestDecodeRefusesArgumentsThatAreNotAnObject(t *testing.T) {
	t.Parallel()

	_, err := toolcall.Decode(`[1,2,3]`)
	require.ErrorIs(t, err, toolcall.ErrNotAnObject)
}

func TestDecodeKeepsFractionsAndStrings(t *testing.T) {
	t.Parallel()

	args, err := toolcall.Decode(`{"ratio":0.5,"name":"x","flag":true,"list":[1,2]}`)
	require.NoError(t, err)
	assert.InDelta(t, 0.5, args["ratio"], 0)
	assert.Equal(t, "x", args["name"])
	assert.Equal(t, true, args["flag"])
	assert.Equal(t, []any{int64(1), int64(2)}, args["list"])
}
