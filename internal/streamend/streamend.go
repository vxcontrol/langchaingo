package streamend

import (
	"context"
	"errors"
	"slices"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
)

func Incomplete(ctx context.Context, readErr error) error {
	errs := []error{llms.ErrIncompleteStream}
	if readErr != nil {
		errs = append(errs, readErr)
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		for _, err := range []error{ctxErr, context.Cause(ctx)} {
			if !slices.ContainsFunc(errs, func(have error) bool { return errors.Is(have, err) }) {
				errs = append(errs, err)
			}
		}
	}
	return &incomplete{errs: errs}
}

type incomplete struct {
	errs []error
}

func (e *incomplete) Error() string {
	texts := make([]string, len(e.errs))
	for i, err := range e.errs {
		texts[i] = err.Error()
	}
	return strings.Join(texts, ": ")
}

func (e *incomplete) Unwrap() []error {
	return e.errs
}
