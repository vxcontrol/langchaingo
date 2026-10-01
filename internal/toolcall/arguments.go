// Package toolcall decodes the arguments a model sent with a tool call.
package toolcall

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// ErrNotAnObject reports arguments that are not a JSON object.
var ErrNotAnObject = errors.New("toolcall: arguments are not a JSON object")

// Field is one argument, carrying the key order the model sent.
type Field struct {
	Key   string
	Value any
}

// DecodeFields decodes tool call arguments, keeping the order of the keys and
// the exact value of integers wider than float64 can hold.
func DecodeFields(raw string) ([]Field, error) {
	if absent(raw) {
		return nil, nil
	}
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()

	open, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if delim, ok := open.(json.Delim); !ok || delim != '{' {
		return nil, fmt.Errorf("%w: %s", ErrNotAnObject, raw)
	}

	var fields []Field
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			return nil, err
		}
		name, ok := key.(string)
		if !ok {
			return nil, fmt.Errorf("%w: %s", ErrNotAnObject, raw)
		}
		var value any
		if err := dec.Decode(&value); err != nil {
			return nil, err
		}
		fields = append(fields, Field{Key: name, Value: exactNumbers(value)})
	}
	if _, err := dec.Token(); err != nil {
		return nil, err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("%w: data follows the object: %s", ErrNotAnObject, raw)
	}

	return fields, nil
}

func Normalize(raw string) string {
	if absent(raw) {
		return "{}"
	}
	return raw
}

func Schema(parameters any) any {
	if parameters == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}
	return parameters
}

func absent(raw string) bool {
	trimmed := strings.TrimSpace(raw)
	return trimmed == "" || trimmed == "null"
}

// Decode decodes tool call arguments into a map, keeping integers exact.
func Decode(raw string) (map[string]any, error) {
	fields, err := DecodeFields(raw)
	if err != nil {
		return nil, err
	}

	arguments := make(map[string]any, len(fields))
	for _, field := range fields {
		arguments[field.Key] = field.Value
	}

	return arguments, nil
}

func exactNumbers(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		for key, item := range typed {
			typed[key] = exactNumbers(item)
		}
	case []any:
		for i, item := range typed {
			typed[i] = exactNumbers(item)
		}
	case json.Number:
		if whole, err := typed.Int64(); err == nil {
			return whole
		}
		if fraction, err := typed.Float64(); err == nil {
			return fraction
		}
	}

	return value
}
