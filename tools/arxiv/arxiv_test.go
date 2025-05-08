package arxiv

import "testing"

func TestNew(t *testing.T) {
	t.Parallel()

	tool, err := New(10, DefaultUserAgent)
	if err != nil {
		t.Fatal(err)
	}
	call, err := tool.Call(t.Context(), "electron")
	if err != nil {
		t.Fatal(err)
	}
	t.Log(call)
}
