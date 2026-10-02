package bedrockclient

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/vxcontrol/langchaingo/internal/awsbody"
)

type legacyRuntime interface {
	InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput,
		optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
	InvokeModelWithResponseStream(ctx context.Context, params *bedrockruntime.InvokeModelWithResponseStreamInput,
		optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelWithResponseStreamOutput, error)
}

type legacyReadOnlyBodies struct{ legacyRuntime }

func (c legacyReadOnlyBodies) InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.InvokeModelOutput, error) {
	return c.legacyRuntime.InvokeModel(ctx, params, append(optFns, awsbody.WithoutWriteTo)...)
}

func (c legacyReadOnlyBodies) InvokeModelWithResponseStream(ctx context.Context,
	params *bedrockruntime.InvokeModelWithResponseStreamInput, optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.InvokeModelWithResponseStreamOutput, error) {
	return c.legacyRuntime.InvokeModelWithResponseStream(ctx, params, append(optFns, awsbody.WithoutWriteTo)...)
}

type converseReadOnlyBodies struct{ BedrockRuntimeClientInterface }

func (c converseReadOnlyBodies) Converse(ctx context.Context, input *bedrockruntime.ConverseInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	return c.BedrockRuntimeClientInterface.Converse(ctx, input, append(optFns, awsbody.WithoutWriteTo)...)
}

func (c converseReadOnlyBodies) ConverseStream(ctx context.Context, input *bedrockruntime.ConverseStreamInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseStreamOutput, error) {
	return c.BedrockRuntimeClientInterface.ConverseStream(ctx, input, append(optFns, awsbody.WithoutWriteTo)...)
}
