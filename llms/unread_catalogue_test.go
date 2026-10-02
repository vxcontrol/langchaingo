package llms_test

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
)

var optionsOutsideTheCatalogue = map[string]bool{
	"Model": true, "MaxTokens": true, "Temperature": true, "StopWords": true, "StreamingFunc": true,
	"TopP": true, "Reasoning": true, "FailOnTruncation": true, "StructuredOutput": true,
	"Tools": true, "ToolChoice": true, "Functions": true, "FunctionCallBehavior": true,
	"ExtraBody": true, "Metadata": true, "Voice": true, "Speed": true, "ResponseFormat": true,
	"WebSearchOptions": true,
}

func setAsked(t *testing.T, field reflect.Value, name string) {
	t.Helper()

	switch field.Interface().(type) {
	case *float64:
		field.Set(reflect.ValueOf(new(0.5)))
	case *int:
		field.Set(reflect.ValueOf(new(7)))
	case *string:
		field.Set(reflect.ValueOf(new("asked")))
	case *bool:
		field.Set(reflect.ValueOf(new(true)))
	case bool:
		field.SetBool(true)
	default:
		t.Fatalf("CallOptions.%s is neither reported by the unread catalogue nor listed outside it", name)
	}
}

func catalogueOptions(t *testing.T) []reflect.StructField {
	t.Helper()

	var fields []reflect.StructField
	for _, field := range reflect.VisibleFields(reflect.TypeFor[llms.CallOptions]()) {
		switch {
		case !field.IsExported():
			t.Errorf("CallOptions.%s is unexported, so no door outside llms can carry it", field.Name)
		case !optionsOutsideTheCatalogue[field.Name]:
			fields = append(fields, field)
		}
	}
	require.NotEmpty(t, fields)
	return fields
}

func TestEveryCallOptionIsReportedByTheUnreadCatalogueOrListedOutsideIt(t *testing.T) {
	t.Parallel()

	for _, field := range catalogueOptions(t) {
		opts := llms.CallOptions{}
		setAsked(t, reflect.ValueOf(&opts).Elem().FieldByIndex(field.Index), field.Name)

		var warn llms.Warnings
		warn.AddUnreadOptions("m", opts, "no field")

		reported := warn.List()
		require.Len(t, reported, 1, "a door that leaves CallOptions.%s off the wire must say so", field.Name)
		require.Equal(t, "With"+field.Name, reported[0].Option)
		require.Equal(t, llms.WarningDrop, reported[0].Kind)
		require.NotEmpty(t, reported[0].Asked, "%s reported without the value asked", field.Name)
	}
}

func TestTheUnreadCatalogueReportsEveryOptionADoorDoesNotCarry(t *testing.T) {
	t.Parallel()

	opts := llms.CallOptions{}
	var want []string
	for _, field := range catalogueOptions(t) {
		setAsked(t, reflect.ValueOf(&opts).Elem().FieldByIndex(field.Index), field.Name)
		if field.Name != "Seed" && field.Name != "TopK" {
			want = append(want, "With"+field.Name)
		}
	}

	var warn llms.Warnings
	warn.AddUnreadOptions("m", opts, "no field", "WithSeed", "WithTopK")

	reported := make([]string, 0, len(want))
	for _, w := range warn.List() {
		reported = append(reported, w.Option)
	}
	require.ElementsMatch(t, want, reported)
}

func TestTheOptionsADoorReshapesAreOutsideTheUnreadCatalogue(t *testing.T) {
	t.Parallel()

	opts := llms.CallOptions{}
	for _, apply := range []llms.CallOption{
		llms.WithTemperature(0.4), llms.WithTopP(0.9), llms.WithMaxTokens(1024),
		llms.WithStopWords([]string{"stop"}),
	} {
		apply(&opts)
	}

	var warn llms.Warnings
	warn.AddUnreadOptions("m", opts, "no field")

	require.Empty(t, warn.List(), "an option the door reshapes is reported with the value that travelled, not as unread")
}

func TestAnExplicitZeroSeedIsReportedNotSwallowed(t *testing.T) {
	t.Parallel()

	opts := llms.CallOptions{}
	llms.WithSeed(0)(&opts)

	var warn llms.Warnings
	warn.AddUnreadOptions("m", opts, "no field")

	reported := make(map[string]llms.Warning)
	for _, w := range warn.List() {
		reported[w.Option] = w
	}
	w, ok := reported["WithSeed"]
	require.True(t, ok, "an explicit seed of 0 is a real seed, not the absence of one")
	require.Equal(t, llms.WarningDrop, w.Kind)
	require.Equal(t, "0", w.Asked)
}

func TestACallThatSetsNoCatalogueOptionReportsNothing(t *testing.T) {
	t.Parallel()

	var warn llms.Warnings
	warn.AddUnreadOptions("m", llms.CallOptions{}, "no field")

	require.Empty(t, warn.List())
}
