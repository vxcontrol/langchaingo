package toolcall

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestProbeUvxTrailing(t *testing.T) {
	for _, in := range []string{`{"a":1}{"b":2}`, `{"city":"Paris"}{"city":"London"}`, `{"a":1} garbage`, `{"a":1}]`, `{"a":1}   `} {
		m, err := Decode(in)
		var u map[string]any
		uerr := json.Unmarshal([]byte(in), &u)
		fmt.Printf("%-40q Decode=%v err=%v | json.Unmarshal err=%v\n", in, m, err, uerr)
	}
}
