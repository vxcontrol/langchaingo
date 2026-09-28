package llms

import "testing"

func TestANilSinkAcceptsWarningsAndKeepsNone(t *testing.T) {
	t.Parallel()

	var sink *Warnings
	sink.Add(Warning{Kind: WarningDrop, Option: "WithTopP"})
	if got := sink.List(); got != nil {
		t.Errorf("a nil sink kept %v", got)
	}
}

func TestASinkKeepsWhatItIsGiven(t *testing.T) {
	t.Parallel()

	sink := &Warnings{}
	sink.Add(Warning{Kind: WarningDrop, Option: "WithTopP", Model: "m", Asked: "0.9"})
	sink.Add(Warning{Kind: WarningClamp, Option: "WithMaxTokens", Model: "m", Asked: "1", Sent: "2"})
	if got := sink.List(); len(got) != 2 || got[0].Option != "WithTopP" || got[1].Sent != "2" {
		t.Errorf("sink kept %v", got)
	}
}
