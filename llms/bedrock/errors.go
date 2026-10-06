package bedrock

import (
	"errors"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
)

// errorMapping represents a mapping from error patterns to error codes.
type errorMapping struct {
	patterns []string
	code     llms.ErrorCode
	message  string
}

// bedrockErrorMappings defines the error mappings for AWS Bedrock.
var bedrockErrorMappings = []errorMapping{
	{
		patterns: []string{"accessdeniedexception", "notauthorized", "optinrequired"},
		code:     llms.ErrCodeAuthentication,
		message:  "Access denied: the identity lacks the IAM permission or the model access",
	},
	{
		patterns: []string{
			"unrecognizedclientexception", "expiredtokenexception", "incompletesignature",
			"unauthorized", "invalid security token",
		},
		code:    llms.ErrCodeAuthentication,
		message: "Invalid, expired or missing AWS credentials",
	},
	{
		patterns: []string{"throttlingexception", "toomanyrequestsexception"},
		code:     llms.ErrCodeRateLimit,
		message:  "Request rate limit exceeded",
	},
	{
		patterns: []string{"resourcenotfoundexception", "model not found"},
		code:     llms.ErrCodeResourceNotFound,
		message:  "Model not found or not accessible",
	},
	{
		patterns: []string{"validationexception", "validationerror", "malformed"},
		code:     llms.ErrCodeInvalidRequest,
		message:  "Invalid request parameters",
	},
	{
		patterns: []string{"modeltimeoutexception", "requesttimeoutexception"},
		code:     llms.ErrCodeTimeout,
		message:  "Model invocation timeout",
	},
	{
		patterns: []string{
			"serviceexception", "internalserverexception", "internalservererror", "internalfailure",
			"serviceunavailable", "statuscode: 500",
		},
		code:    llms.ErrCodeProviderUnavailable,
		message: "AWS Bedrock service error",
	},
	{
		patterns: []string{"modelnotreadyexception"},
		code:     llms.ErrCodeProviderUnavailable,
		message:  "Model not ready for invocation",
	},
	{
		patterns: []string{"payload size", "token limit", "requestentitytoolarge"},
		code:     llms.ErrCodeTokenLimit,
		message:  "Input size or token limit exceeded",
	},
}

// MapError maps AWS Bedrock-specific errors to standardized error codes. An
// error that already is an *llms.Error, such as ErrCodeTruncated from
// llms.WithFailOnTruncation, comes back unchanged.
func MapError(err error) error {
	if err == nil {
		return nil
	}
	var typed *llms.Error
	if errors.As(err, &typed) {
		return err
	}

	errStr := strings.ToLower(err.Error())

	// Check each error mapping
	for _, mapping := range bedrockErrorMappings {
		for _, pattern := range mapping.patterns {
			if strings.Contains(errStr, pattern) {
				return llms.NewError(mapping.code, "bedrock", mapping.message).WithCause(err)
			}
		}
	}

	// Use the generic error mapper for unrecognized errors
	mapper := llms.NewErrorMapper("bedrock")
	return mapper.Map(err)
}
