package bedrockclient

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	bedrockClient := &bedrockruntime.Client{}
	client := NewClient(bedrockClient)

	require.NotNil(t, client)
	assert.Equal(t, bedrockClient, client.client)
}
