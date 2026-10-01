package bedrockclient

import (
	"context"
	"errors"
	"fmt"

	"github.com/vxcontrol/langchaingo/internal/streamend"
	"github.com/vxcontrol/langchaingo/llms"

	"github.com/aws/smithy-go"
)

func streamEndError(ctx context.Context, finished bool, err error) error {
	if err != nil {
		var apiErr smithy.APIError
		if errors.As(err, &apiErr) {
			return fmt.Errorf("%w: stream error: %w", llms.ErrStreamFailed, err)
		}
		if finished {
			return nil
		}
		return streamend.Incomplete(ctx, fmt.Errorf("stream error: %w", err))
	}
	if finished {
		return nil
	}
	return streamend.Incomplete(ctx, nil)
}
