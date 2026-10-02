package awsbody

import (
	"io"
	"net/http"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

func WithoutWriteTo(o *bedrockruntime.Options) {
	if o.HTTPClient != nil {
		o.HTTPClient = readOnlyBody{o.HTTPClient}
	}
}

type readOnlyBody struct{ client bedrockruntime.HTTPClient }

func (c readOnlyBody) Do(req *http.Request) (*http.Response, error) {
	if req.Body != nil && req.Body != http.NoBody {
		req.Body = struct{ io.ReadCloser }{req.Body}
	}
	return c.client.Do(req)
}
