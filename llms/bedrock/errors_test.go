package bedrock_test

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestMapErrorTellsMissingPermissionFromBadCredentials(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ name, sdk, message string }{
		{"missing permission", "operation error Bedrock Runtime: Converse, AccessDeniedException: not authorized to invoke",
			"Access denied: the identity lacks the IAM permission or the model access"},
		{"unknown access key", "operation error Bedrock Runtime: Converse, UnrecognizedClientException: The security token included in the request is invalid",
			"Invalid or missing AWS credentials"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var mapped *llms.Error
			require.ErrorAs(t, bedrock.MapError(errors.New(tc.sdk)), &mapped)
			require.Equal(t, llms.ErrCodeAuthentication, mapped.Code)
			require.Equal(t, tc.message, mapped.Message)
		})
	}
}
