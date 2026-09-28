package anthropicclient

import (
	"context"
	"net/http"
	"testing"
)

func TestProbeV3FedLifetime(t *testing.T) {
	for _, life := range []int{60, 61, 90, 600} {
		s := newFederationStub(t)
		s.lifetime = life
		_, auth := s.client(t, FederationConfig{RuleID: "r", OrganizationID: "o", ServiceAccountID: "s", Assertion: AssertionFromString("jwt1")})
		for i := 0; i < 3; i++ {
			if _, err := auth.accessToken(context.Background()); err != nil {
				t.Fatal(err)
			}
		}
		n, _, _, _ := s.counts()
		t.Logf("expires_in=%d: 3 token requests -> %d exchanges", life, n)
	}
	// jti single-use: the server rejects a second exchange of the same unrotated JWT.
	s := newFederationStub(t)
	s.lifetime = 60
	_, auth := s.client(t, FederationConfig{RuleID: "r", OrganizationID: "o", ServiceAccountID: "s", Assertion: AssertionFromString("jwt1")})
	_, err1 := auth.accessToken(context.Background())
	s.mu.Lock()
	s.exchangeStatus = http.StatusUnauthorized // jti_reused
	s.mu.Unlock()
	_, err2 := auth.accessToken(context.Background())
	t.Logf("expires_in=60 jti reuse: first err=%v second err=%v (cached token still has ~60s)", err1, err2)
}
