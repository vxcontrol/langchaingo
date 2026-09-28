package toolcall

import (
	"encoding/json"
	"testing"
)

func TestProbeNull(t *testing.T) {
	for _, raw := range []string{"null", "{}", ""} {
		m, err := Decode(raw)
		var base map[string]any
		berr := json.Unmarshal([]byte(raw), &base)
		t.Logf("raw=%q HEAD Decode -> %v err=%v | json.Unmarshal (base) -> %v err=%v", raw, m, err, base, berr)
	}
}
