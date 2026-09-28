package pgvector

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestProbeVfy3PgFloat(t *testing.T) {
	for _, body := range []string{`{"user_id": 1234567}`, `{"user_id": 42}`, `{"price": 0.00001}`, `{"ts": 1727500000}`} {
		var filter map[string]any
		_ = json.Unmarshal([]byte(body), &filter)
		for k, v := range filter {
			_, args, err := filterPredicates("", filter, 0)
			stored, _ := json.Marshal(map[string]any{k: v}) // what pgx writes into the json column
			var raw map[string]json.RawMessage
			_ = json.Unmarshal(stored, &raw)
			fmt.Printf("filter %s: bound=%q  stored(->>)=%s  match=%v err=%v\n", body, args[0], raw[k], args[0] == string(raw[k]), err)
		}
	}
	_, args, _ := filterPredicates("", map[string]any{"user_id": 1234567}, 0)
	fmt.Printf("int filter bound=%q\n", args[0])
}
