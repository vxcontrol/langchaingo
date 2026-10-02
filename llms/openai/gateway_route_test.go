package openai

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// gatewayRoute sends a recording through an OpenAI-compatible gateway route.
// The cassette keeps the request the door made; the response is the gateway's,
// not the vendor's.
type gatewayRoute struct {
	base        string
	modelPrefix string
}

func (g gatewayRoute) RoundTrip(req *http.Request) (*http.Response, error) {
	base, err := url.Parse(g.base)
	if err != nil {
		return nil, err
	}
	out := req.Clone(req.Context())
	out.URL.Scheme, out.URL.Host, out.Host = base.Scheme, base.Host, ""
	out.URL.Path = strings.TrimSuffix(base.Path, "/") + strings.TrimPrefix(req.URL.Path, "/v1")
	if req.Body != nil {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		var fields map[string]any
		if json.Unmarshal(body, &fields) == nil {
			if model, ok := fields["model"].(string); ok && !strings.HasPrefix(model, g.modelPrefix) {
				fields["model"] = g.modelPrefix + model
				if body, err = json.Marshal(fields); err != nil {
					return nil, err
				}
			}
		}
		out.Body = io.NopCloser(bytes.NewReader(body))
		out.ContentLength = int64(len(body))
	}
	return http.DefaultTransport.RoundTrip(out)
}
