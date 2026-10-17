package service

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	coderws "github.com/coder/websocket"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestDeepSeekWSCompactionWritesExactlyOneCompactionItem(t *testing.T) {
	gin.SetMode(gin.TestMode)

	payload := mustMarshalDeepSeekCompactTestJSON(t, map[string]any{
		"type":  "response.create",
		"model": "gpt-5.5",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "Continue the task."},
			map[string]any{"type": "compaction_trigger"},
		},
	})
	upstream := &httpUpstreamRecorder{resp: deepSeekRemoteCompactResponsesResponse(deepSeekRemoteCompactTestSummary)}
	svc := &OpenAIGatewayService{cfg: deepSeekMappedResponsesTestConfig(), httpUpstream: upstream}
	account := openaiMappedDeepSeekResponsesAccount()

	c, _ := deepSeekResponsesTestContext(t, payload)
	var events [][]byte
	result, err := svc.proxyOpenAIWSHTTPBridgeTurn(
		deepSeekCompactTestContext(42), c, account, "sk-test", payload, len(payload),
		"gpt-5.5", "", "", "", "", 1,
		func(message []byte) error {
			events = append(events, append([]byte(nil), message...))
			return nil
		},
	)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "deepseek-flash", gjson.GetBytes(upstream.lastBody, "model").String())
	require.False(t, HasCompactionTriggerInInput(upstream.lastBody))
	require.Equal(t, deepSeekCompactInstruction, gjson.GetBytes(upstream.lastBody, "input.1.content.0.text").String())
	require.Len(t, events, 2)
	require.Equal(t, "response.output_item.done", gjson.GetBytes(events[0], "type").String())
	require.Equal(t, "response.completed", gjson.GetBytes(events[1], "type").String())
	outputs := gjson.GetBytes(events[1], "response.output").Array()
	require.Len(t, outputs, 1)
	require.Equal(t, "compaction", outputs[0].Get("type").String())
	require.NotEmpty(t, outputs[0].Get("encrypted_content").String())
	require.NotEmpty(t, result.wsDeepSeekCompactionCheckpoint)
}

func TestDeepSeekWSNextTurnRestoresOwnedCheckpoint(t *testing.T) {
	gin.SetMode(gin.TestMode)

	compactPayload := mustMarshalDeepSeekCompactTestJSON(t, map[string]any{
		"type":  "response.create",
		"model": "gpt-5.5",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "First turn."},
			map[string]any{"type": "compaction_trigger"},
		},
	})
	compactUpstream := &httpUpstreamRecorder{resp: deepSeekRemoteCompactResponsesResponse(deepSeekRemoteCompactTestSummary)}
	svc := &OpenAIGatewayService{cfg: deepSeekMappedResponsesTestConfig(), httpUpstream: compactUpstream}
	account := openaiMappedDeepSeekResponsesAccount()
	c, _ := deepSeekResponsesTestContext(t, compactPayload)

	compacted, err := svc.proxyOpenAIWSHTTPBridgeTurn(
		deepSeekCompactTestContext(42), c, account, "sk-test", compactPayload, len(compactPayload),
		"gpt-5.5", "", "", "", "", 1, func([]byte) error { return nil },
	)
	require.NoError(t, err)
	require.NotEmpty(t, compacted.wsDeepSeekCompactionCheckpoint)

	nextPayload := mustMarshalDeepSeekCompactTestJSON(t, map[string]any{
		"type":  "response.create",
		"model": "gpt-5.5",
		"input": []any{
			compacted.wsDeepSeekCompactionCheckpoint,
			map[string]any{"type": "message", "role": "user", "content": "Continue after compaction."},
		},
	})
	nextUpstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body: io.NopCloser(strings.NewReader(deepSeekRemoteCompactSSEWire(
			`{"type":"response.completed","response":{"id":"resp_after_compact","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`,
		))),
	}}
	svc.httpUpstream = nextUpstream
	nextCtx, _ := deepSeekResponsesTestContext(t, nextPayload)
	_, err = svc.proxyOpenAIWSHTTPBridgeTurn(
		deepSeekCompactTestContext(42), nextCtx, account, "sk-test", nextPayload, len(nextPayload),
		"gpt-5.5", "", "", "", "", 2, func([]byte) error { return nil },
	)
	require.NoError(t, err)
	require.Equal(t, "message", gjson.GetBytes(nextUpstream.lastBody, "input.0.type").String())
	require.Equal(t, "user", gjson.GetBytes(nextUpstream.lastBody, "input.0.role").String())
	require.Contains(t, gjson.GetBytes(nextUpstream.lastBody, "input.0.content.0.text").String(), deepSeekRemoteCompactTestSummary)
	require.False(t, gjson.GetBytes(nextUpstream.lastBody, `input.#(type=="compaction")`).Exists())
}

func TestDeepSeekWSCompactionBridgeOnlyTouchesGPT55(t *testing.T) {
	gin.SetMode(gin.TestMode)

	account := openaiMappedDeepSeekResponsesAccount()
	account.Credentials["model_mapping"] = map[string]any{
		"gpt-5.5":     "deepseek-flash",
		"gpt-6.1-sol": "deepseek-flash",
	}
	require.True(t, shouldForceDeepSeekCompactWSBridge(account, []byte(`{"model":"gpt-5.5"}`)))
	require.False(t, shouldForceDeepSeekCompactWSBridge(account, []byte(`{"model":"gpt-6.1-sol"}`)))
}

func TestDeepSeekWSCompactionEventsSurviveWebSocket(t *testing.T) {
	gin.SetMode(gin.TestMode)

	payload := mustMarshalDeepSeekCompactTestJSON(t, map[string]any{
		"type":  "response.create",
		"model": "gpt-5.5",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "Continue the task."},
			map[string]any{"type": "compaction_trigger"},
		},
	})
	upstream := &httpUpstreamRecorder{resp: deepSeekRemoteCompactResponsesResponse(deepSeekRemoteCompactTestSummary)}
	svc := &OpenAIGatewayService{
		cfg: &config.Config{
			Security: config.SecurityConfig{
				URLAllowlist: config.URLAllowlistConfig{
					Enabled:           false,
					AllowInsecureHTTP: true,
				},
			},
			JWT: config.JWTConfig{Secret: "deepseek-compact-test-jwt-secret-32-bytes"},
			Gateway: config.GatewayConfig{
				MaxLineSize: defaultMaxLineSize,
			},
		},
		httpUpstream: upstream,
	}
	account := openaiMappedDeepSeekResponsesAccount()

	resultCh := make(chan error, 1)
	wsServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := coderws.Accept(w, r, nil)
		if err != nil {
			resultCh <- err
			return
		}
		defer func() { _ = conn.CloseNow() }()
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = r.Clone(r.Context())
		_, bridgeErr := svc.proxyOpenAIWSHTTPBridgeTurn(
			deepSeekCompactTestContext(42), c, account, "sk-test", payload, len(payload),
			"gpt-5.5", "", "", "", "", 1,
			func(message []byte) error {
				writeCtx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
				defer cancel()
				return conn.Write(writeCtx, coderws.MessageText, message)
			},
		)
		resultCh <- bridgeErr
	}))
	defer wsServer.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	client, _, err := coderws.Dial(ctx, "ws"+strings.TrimPrefix(wsServer.URL, "http"), nil)
	cancel()
	require.NoError(t, err)
	defer func() { _ = client.CloseNow() }()

	readEvent := func() []byte {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_, message, readErr := client.Read(ctx)
		require.NoError(t, readErr)
		return message
	}
	output := readEvent()
	completed := readEvent()
	require.Equal(t, "response.output_item.done", gjson.GetBytes(output, "type").String())
	require.Equal(t, "compaction", gjson.GetBytes(output, "item.type").String())
	require.Equal(t, "response.completed", gjson.GetBytes(completed, "type").String())
	require.Len(t, gjson.GetBytes(completed, "response.output").Array(), 1)
	require.Equal(t, "compaction", gjson.GetBytes(completed, "response.output.0.type").String())
	select {
	case bridgeErr := <-resultCh:
		require.NoError(t, bridgeErr)
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for websocket bridge")
	}
}
