package openaiclient

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// GetJSON decodes the JSON the API answers to a GET on suffix, a path under the
// base URL.
func (c *Client) GetJSON(ctx context.Context, suffix string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.buildURL(suffix, ""), http.NoBody)
	if err != nil {
		return err
	}
	c.setHeaders(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return sanitizeHTTPError(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", suffix, resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}
