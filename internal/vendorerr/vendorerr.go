// Package vendorerr types the refusals a vendor documents for the history a
// request replays and for its size.
package vendorerr

import (
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/vxcontrol/langchaingo/llms"
)

var blockPosition = regexp.MustCompile(`messages\.(\d+)\.content\.\d+`)

// Classify returns err inside the llms type for the refusal the vendor's message
// names, or err itself. status is the HTTP status of the answer, 0 when unknown.
// origins[i] is the index, in the caller's messages, of the i-th message the
// door sent; nil when the door did not build the messages the vendor read.
func Classify(err error, status int, message string, origins []int) error {
	if status == http.StatusRequestEntityTooLarge || strings.Contains(message, "prompt is too long") {
		return &llms.ErrContextOverflow{Cause: err}
	}
	fault, ok := historyFault(message)
	if !ok {
		return err
	}
	return &llms.ErrHistoryRejected{Fault: fault, Message: origin(message, origins), Cause: err}
}

func historyFault(message string) (llms.HistoryFault, bool) {
	switch {
	case strings.Contains(message, "`thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified"):
		return llms.HistoryThinkingModified, true
	case strings.Contains(message, "Invalid `signature` in `thinking` block. The block is bound to a different conversation"):
		return llms.HistoryPrefixChanged, true
	case strings.Contains(message, "Invalid `signature` in `thinking` block"):
		return llms.HistorySignatureInvalid, true
	}
	return "", false
}

func origin(message string, origins []int) int {
	match := blockPosition.FindStringSubmatch(message)
	if match == nil {
		return -1
	}
	i, err := strconv.Atoi(match[1])
	if err != nil || i >= len(origins) {
		return -1
	}
	return origins[i]
}
