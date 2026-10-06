package bedrock_test

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/bedrock"
)

func TestMapErrorReadsTheCommonErrorsAWSDocuments(t *testing.T) {
	t.Parallel()

	const (
		accessDenied = "Access denied: the identity lacks the IAM permission or the model access"
		credentials  = "Invalid, expired or missing AWS credentials"
		unavailable  = "AWS Bedrock service error"
	)
	for _, tc := range []struct {
		name, sdk string
		code      llms.ErrorCode
		message   string
	}{
		{"missing permission", "StatusCode: 403, RequestID: 1f2e, AccessDeniedException: not authorized to invoke",
			llms.ErrCodeAuthentication, accessDenied},
		{"not authorized", "StatusCode: 401, RequestID: 1f2e, NotAuthorized: You don't have permissions",
			llms.ErrCodeAuthentication, accessDenied},
		{"no subscription", "StatusCode: 403, RequestID: 1f2e, OptInRequired: Your AWS account needs a subscription",
			llms.ErrCodeAuthentication, accessDenied},
		{"unknown access key", "StatusCode: 403, RequestID: 1f2e, UnrecognizedClientException: The security token included in the request is invalid",
			llms.ErrCodeAuthentication, credentials},
		{"expired session", "StatusCode: 403, RequestID: 6f1b-5004, ExpiredTokenException: The security token included in the request has expired",
			llms.ErrCodeAuthentication, credentials},
		{"bad signature", "StatusCode: 403, RequestID: 1f2e, IncompleteSignature: The request signature doesn't conform to AWS standards",
			llms.ErrCodeAuthentication, credentials},
		{"internal failure", "StatusCode: 500, RequestID: 1f2e, InternalFailure: The request can't be processed right now",
			llms.ErrCodeProviderUnavailable, unavailable},
		{"internal server", "StatusCode: 500, RequestID: 1f2e, InternalServerException: internal error",
			llms.ErrCodeProviderUnavailable, unavailable},
		{"unavailable", "StatusCode: 503, RequestID: 1f2e, ServiceUnavailableException: try again later",
			llms.ErrCodeProviderUnavailable, unavailable},
		{"request timeout", "StatusCode: 408, RequestID: 1f2e, RequestTimeoutException: The request timed out",
			llms.ErrCodeTimeout, "Model invocation timeout"},
		{"validation error", "StatusCode: 400, RequestID: 1f2e, ValidationError: The input doesn't meet the required format",
			llms.ErrCodeInvalidRequest, "Invalid request parameters"},
		{"entity too large", "StatusCode: 413, RequestID: 1f2e, RequestEntityTooLargeException: The request entity is too large",
			llms.ErrCodeTokenLimit, "Input size or token limit exceeded"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var mapped *llms.Error
			require.ErrorAs(t, bedrock.MapError(errors.New("operation error Bedrock Runtime: Converse, https response error "+tc.sdk)), &mapped)
			require.Equal(t, tc.code, mapped.Code)
			require.Equal(t, tc.message, mapped.Message)
		})
	}
}

func TestMapErrorDoesNotReadTheRequestIDAsAStatus(t *testing.T) {
	t.Parallel()

	var mapped *llms.Error
	require.ErrorAs(t, bedrock.MapError(errors.New("operation error Bedrock Runtime: Converse, https response error "+
		"StatusCode: 404, RequestID: 6f1b-5004, UnknownOperationException: The action isn't recognized")), &mapped)
	require.NotEqual(t, llms.ErrCodeProviderUnavailable, mapped.Code)
}

func TestMapErrorKeepsAnErrorThatAlreadyCarriesACode(t *testing.T) {
	t.Parallel()

	truncated := llms.CheckTruncation(&llms.ContentResponse{Choices: []*llms.ContentChoice{{StopReason: "max_tokens"}}},
		llms.CallOptions{FailOnTruncation: true})
	require.True(t, llms.IsTruncatedError(bedrock.MapError(truncated)))
}
