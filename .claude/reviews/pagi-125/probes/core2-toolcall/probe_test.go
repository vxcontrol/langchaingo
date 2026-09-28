package toolcall_test

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vxcontrol/langchaingo/internal/toolcall"
)

func TestProbeTrailing(t *testing.T) {
	for _, raw := range []string{`{"a":1}{"b":2}`, `{"a":1} trailing`, `{"a":1}]`, `{"city":"Paris"}{"city":"London"}`, ` {"a":1} `, `null`, ``} {
		m, err := toolcall.Decode(raw)
		var base map[string]any
		berr := json.Unmarshal([]byte(raw), &base)
		fmt.Printf("PROBE %-40q head=%v err=%v | base(json.Unmarshal)=%v err=%v\n", raw, m, err, base, berr)
	}
}
