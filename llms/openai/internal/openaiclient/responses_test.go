package openaiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vxcontrol/langchaingo/llms"
	"github.com/vxcontrol/langchaingo/llms/streaming"
)

const responsesAnswer = `{"id":"resp_1","object":"response","model":"gpt-6-luna","status":"completed","error":null,
"incomplete_details":null,"output":[
{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Scan first."}],"encrypted_content":"enc-1"},
{"type":"message","id":"msg_1","status":"completed","role":"assistant","phase":"commentary",
 "content":[{"type":"output_text","text":"Starting a scan.","annotations":[],"logprobs":[]}]},
{"type":"function_call","id":"fc_1","call_id":"call_1","name":"nmap","arguments":"{\"host\":\"A\"}","status":"completed"}],
"usage":{"input_tokens":81,"input_tokens_details":{"cached_tokens":64,"cache_write_tokens":8},
"output_tokens":1035,"output_tokens_details":{"reasoning_tokens":832},"total_tokens":1116}}`

func responsesServer(t *testing.T, status int, body string) (*Client, *[]byte, *string) {
	t.Helper()

	var sent []byte
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sent, _ = io.ReadAll(r.Body)
		path = r.URL.Path
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	client, err := New("token", "gpt-6-luna", srv.URL+"/v1", "", APITypeOpenAI, "", http.DefaultClient, "", nil, false, false, false)
	require.NoError(t, err)
	return client, &sent, &path
}

func TestAResponseComesBackWithItsItemsAndUsage(t *testing.T) {
	t.Parallel()

	client, sent, path := responsesServer(t, http.StatusOK, responsesAnswer)
	resp, err := client.CreateResponse(t.Context(), &ResponsesRequest{
		Model: "gpt-6-luna", Input: []any{ResponsesMessage{Type: "message", Role: "user", Content: "Scan host A."}},
		Metadata: map[string]any{"openai:internal": true, "flow": "7"},
	})
	require.NoError(t, err)

	require.Equal(t, "/v1/responses", *path)
	var body map[string]any
	require.NoError(t, json.Unmarshal(*sent, &body))
	require.Equal(t, false, body["store"])
	require.NotContains(t, body, "stream")
	require.Equal(t, map[string]any{"flow": "7"}, body["metadata"])

	require.Equal(t, "completed", resp.Status)
	require.Len(t, resp.Output, 3)
	require.Equal(t, "enc-1", resp.Output[0].EncryptedContent)
	require.Equal(t, "commentary", resp.Output[1].Phase)
	require.Equal(t, "call_1", resp.Output[2].CallID)
	require.Equal(t, 64, resp.Usage.InputTokensDetails.CachedTokens)
	require.Equal(t, 8, resp.Usage.InputTokensDetails.CacheWriteTokens)
	require.Equal(t, 832, resp.Usage.OutputTokensDetails.ReasoningTokens)
}

func responsesStream(events ...string) string {
	var b strings.Builder
	for _, event := range events {
		var head struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(event), &head)
		var line bytes.Buffer
		_ = json.Compact(&line, []byte(event))
		b.WriteString("event: " + head.Type + "\ndata: " + line.String() + "\n\n")
	}
	return b.String()
}

func TestAStreamedResponseReachesTheCallbackAndEndsWithTheFinalResponse(t *testing.T) {
	t.Parallel()

	client, sent, _ := responsesServer(t, http.StatusOK, responsesStream(
		`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","status":"in_progress","output":[],"usage":null}}`,
		`{"type":"response.reasoning_summary_text.delta","sequence_number":1,"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"Scan first."}`,
		`{"type":"response.output_text.delta","sequence_number":2,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"Starting "}`,
		`{"type":"response.output_text.delta","sequence_number":3,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"a scan."}`,
		`{"type":"response.output_item.done","sequence_number":4,"output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"nmap","arguments":"{\"host\":\"A\"}","status":"completed"}}`,
		`{"type":"response.completed","sequence_number":5,"response":`+responsesAnswer+`}`,
	))

	var chunks []streaming.Chunk
	resp, err := client.CreateResponse(t.Context(), &ResponsesRequest{
		Model: "gpt-6-luna", Input: []any{},
		StreamingFunc: func(_ context.Context, chunk streaming.Chunk) error {
			chunks = append(chunks, chunk)
			return nil
		},
	})
	require.NoError(t, err)

	var body map[string]any
	require.NoError(t, json.Unmarshal(*sent, &body))
	require.Equal(t, true, body["stream"])
	require.Len(t, chunks, 5)
	require.Equal(t, "Scan first.", chunks[0].Reasoning.Content)
	require.Equal(t, "Starting ", chunks[1].Content)
	require.Equal(t, "a scan.", chunks[2].Content)
	require.Equal(t, streaming.NewToolCall("call_1", "nmap", `{"host":"A"}`), chunks[3].ToolCall)
	require.Equal(t, streaming.ChunkTypeDone, chunks[4].Type)
	require.Len(t, resp.Output, 3)
	require.Equal(t, 1116, resp.Usage.TotalTokens)
}

func TestAStreamWithoutAFinalEventIsIncomplete(t *testing.T) {
	t.Parallel()

	client, _, _ := responsesServer(t, http.StatusOK, responsesStream(
		`{"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"Star"}`,
	))
	var last streaming.Chunk
	_, err := client.CreateResponse(t.Context(), &ResponsesRequest{
		Model: "gpt-6-luna", StreamingFunc: func(_ context.Context, chunk streaming.Chunk) error {
			last = chunk
			return nil
		},
	})
	require.ErrorIs(t, err, llms.ErrIncompleteStream)
	require.Equal(t, streaming.ChunkTypeDone, last.Type, "a cut stream still ends")
}

func TestAFailedResponseIsAnError(t *testing.T) {
	t.Parallel()

	failed := `{"id":"resp_1","status":"failed","error":{"code":"server_error","message":"The model failed to generate a response."},"output":[]}`
	client, _, _ := responsesServer(t, http.StatusOK, failed)
	_, err := client.CreateResponse(t.Context(), &ResponsesRequest{Model: "gpt-6-luna"})
	require.ErrorIs(t, err, llms.ErrStreamFailed)
	require.ErrorContains(t, err, "The model failed to generate a response.")

	client, _, _ = responsesServer(t, http.StatusOK, responsesStream(`{"type":"response.failed","sequence_number":1,"response":`+failed+`}`))
	_, err = client.CreateResponse(t.Context(), &ResponsesRequest{
		Model: "gpt-6-luna", StreamingFunc: func(context.Context, streaming.Chunk) error { return nil },
	})
	require.ErrorIs(t, err, llms.ErrStreamFailed)

	client, _, _ = responsesServer(t, http.StatusOK, responsesStream(`{"type":"error","sequence_number":1,"code":"server_error","message":"overloaded"}`))
	_, err = client.CreateResponse(t.Context(), &ResponsesRequest{
		Model: "gpt-6-luna", StreamingFunc: func(context.Context, streaming.Chunk) error { return nil },
	})
	require.ErrorIs(t, err, llms.ErrStreamFailed)
	require.ErrorContains(t, err, "overloaded")
}

func TestAnErrorStatusFromResponsesKeepsTheVendorMessage(t *testing.T) {
	t.Parallel()

	client, _, _ := responsesServer(t, http.StatusBadRequest, `{"error":{"message":"Unsupported parameter: 'stop'.","type":"invalid_request_error"}}`)
	_, err := client.CreateResponse(t.Context(), &ResponsesRequest{Model: "gpt-6-luna"})
	var status *StatusError
	require.ErrorAs(t, err, &status)
	require.Equal(t, http.StatusBadRequest, status.StatusCode)
	require.Equal(t, "Unsupported parameter: 'stop'.", status.Message)
}
