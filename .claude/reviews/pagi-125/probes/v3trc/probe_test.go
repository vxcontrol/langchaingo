package llms

import "testing"

func TestProbeV3Trc(t *testing.T) {
	for _, r := range []string{"model_context_window_exceeded", "max_tokens", "model_length"} {
		resp := &ContentResponse{Choices: []*ContentChoice{{StopReason: r, Truncated: IsTruncated(r)}}}
		t.Logf("%s: IsTruncated=%v CheckTruncation(fail)=%v", r, IsTruncated(r), CheckTruncation(resp, CallOptions{FailOnTruncation: true}))
	}
}
