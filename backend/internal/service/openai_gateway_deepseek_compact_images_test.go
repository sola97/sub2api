package service

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

const compactTestImageURL = "data:image/png;base64,AQID"

func compactVisionResponse(status string, content string) *http.Response {
	body := `{"status":"` + status + `","output":[{"type":"message","role":"assistant","content":` + content + `}]}`
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(body))}
}

func compactImagesService() (*OpenAIGatewayService, *httpUpstreamRecorder) {
	cfg := deepSeekMappedResponsesTestConfig()
	cfg.Gateway.DeepSeekCompactVisionAccountID = 4
	cfg.Gateway.DeepSeekCompactVisionModel = "gpt-6.1-sol"
	vision := Account{ID: 4, Platform: PlatformOpenAI, Type: AccountTypeAPIKey, Status: StatusActive, Schedulable: true, Credentials: map[string]any{"base_url": "https://vision.example/v1", "api_key": "vision-secret"}}
	upstream := &httpUpstreamRecorder{}
	return &OpenAIGatewayService{cfg: cfg, accountRepo: stubOpenAIAccountRepo{accounts: []Account{vision}}, httpUpstream: upstream}, upstream
}

func compactImageHistory(part string) []byte {
	return []byte(`{"model":"gpt-5.6-sol","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"remember the screenshot"},` + part + `]},{"type":"compaction_trigger"}]}`)
}

func TestDeepSeekCompactImagesRoundTripAndSecondCompaction(t *testing.T) {
	svc, upstream := compactImagesService()
	upstream.resp = compactVisionResponse("completed", `[{"type":"output_text","text":"Menu: Token usage, automatic message history."}]`)
	original := compactImageHistory(`{"type":"input_image","image_url":"` + compactTestImageURL + `","detail":"high"}`)
	before := string(original)
	textBody, images, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), original, "deepseek-flash")
	require.NoError(t, err)
	require.Equal(t, before, string(original))
	require.Len(t, images, 1)
	require.NotContains(t, string(textBody), compactTestImageURL)
	require.Contains(t, string(textBody), "Token usage")
	require.Equal(t, "Bearer vision-secret", upstream.requests[0].Header.Get("Authorization"))
	require.Equal(t, "vision.example", upstream.requests[0].URL.Host)
	require.Equal(t, "gpt-6.1-sol", gjson.GetBytes(upstream.bodies[0], "model").String())
	require.Equal(t, compactTestImageURL, gjson.GetBytes(upstream.bodies[0], "input.0.content.1.image_url").String())
	encrypted, err := svc.sealDeepSeekCompactState(deepSeekCompactTestContext(42), deepSeekCompactEnvelope{Checkpoint: "A summary which omits images.", Images: images})
	require.NoError(t, err)
	state, err := svc.openDeepSeekCompactState(deepSeekCompactTestContext(42), encrypted)
	require.NoError(t, err)
	require.Equal(t, deepSeekCompactImageVersion, state.Version)
	require.Empty(t, state.Images[0].Original)
	require.Equal(t, images[0].ID, state.Images[0].ID)
	require.Equal(t, images[0].Description, state.Images[0].Description)
	_, err = svc.openDeepSeekCompactState(deepSeekCompactTestContext(43), encrypted)
	require.ErrorIs(t, err, ErrDeepSeekCompactInvalidEncryptedContent)
	replay := mustMarshalDeepSeekCompactTestJSON(t, map[string]any{"model": "gpt-5.6-sol", "input": []any{map[string]any{"type": "compaction", "encrypted_content": encrypted}, map[string]any{"type": "message", "role": "user", "content": "continue"}}})
	restored, changed, err := svc.RestoreDeepSeekCompactInput(deepSeekCompactTestContext(42), replay)
	require.NoError(t, err)
	require.True(t, changed)
	require.Contains(t, string(restored), "Token usage")
	require.NotContains(t, string(restored), compactTestImageURL)
	var request map[string]any
	require.NoError(t, json.Unmarshal(replay, &request))
	request["input"] = append(request["input"].([]any), map[string]any{"type": "compaction_trigger"})
	second, err := json.Marshal(request)
	require.NoError(t, err)
	_, nextImages, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), second, "deepseek-flash")
	require.NoError(t, err)
	require.Len(t, nextImages, 1)
	require.Equal(t, images[0].ID, nextImages[0].ID)
	require.Equal(t, images[0].Description, nextImages[0].Description)
	require.Len(t, upstream.requests, 1, "old images must not be described again")
	legacy, err := svc.sealDeepSeekCompactCheckpoint(deepSeekCompactTestContext(42), "old text checkpoint")
	require.NoError(t, err)
	legacyState, err := svc.openDeepSeekCompactState(deepSeekCompactTestContext(42), legacy)
	require.NoError(t, err)
	require.Equal(t, 1, legacyState.Version)
	tampered := encrypted[:len(encrypted)-5] + "AAAAA"
	_, err = svc.openDeepSeekCompactState(deepSeekCompactTestContext(42), tampered)
	require.Error(t, err)
}

func TestDeepSeekCompactImagesToolOutputForms(t *testing.T) {
	for _, kind := range []string{"function", "custom_tool", "tool_search"} {
		for _, encoded := range []bool{false, true} {
			t.Run(kind+map[bool]string{false: "array", true: "string"}[encoded], func(t *testing.T) {
				svc, upstream := compactImagesService()
				upstream.resp = compactVisionResponse("completed", `[{"type":"output_text","text":"Screenshot description."}]`)
				parts := []any{map[string]any{"type": "input_image", "image_url": compactTestImageURL}}
				var output any = parts
				if encoded {
					data, _ := json.Marshal(parts)
					output = string(data)
				}
				outputType := kind + "_call_output"
				if kind == "tool_search" {
					outputType = "tool_search_output"
				}
				body := mustMarshalDeepSeekCompactTestJSON(t, map[string]any{"model": "gpt-5.6-sol", "input": []any{map[string]any{"type": kind + "_call", "call_id": "image", "name": "view_image", "arguments": "{}"}, map[string]any{"type": outputType, "call_id": "image", "output": output}, map[string]any{"type": "compaction_trigger"}}})
				text, images, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), body, "deepseek-flash")
				require.NoError(t, err)
				require.Len(t, images, 1)
				require.NotContains(t, string(text), compactTestImageURL)
			})
		}
	}
}

func TestDeepSeekCompactImagesRejectBeforeBillableCalls(t *testing.T) {
	for name, part := range map[string]string{
		"file_only": `{"type":"input_image","file_id":"file-test"}`,
		"bad_data":  `{"type":"input_image","image_url":"data:image/png;base64,invalid!"}`,
	} {
		t.Run(name, func(t *testing.T) {
			svc, upstream := compactImagesService()
			_, _, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), compactImageHistory(part), "deepseek-flash")
			require.Error(t, err)
			require.Empty(t, upstream.requests)
		})
	}
	t.Run("source_limit", func(t *testing.T) {
		svc, upstream := compactImagesService()
		part := `{"type":"input_image","image_url":"https://example.test/` + strings.Repeat("a", compactImageMaxSourceBytes) + `"}`
		_, _, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), compactImageHistory(part), "deepseek-flash")
		require.ErrorContains(t, err, "image source exceeds")
		require.Empty(t, upstream.requests)
	})
	svc, upstream := compactImagesService()
	body := []byte(strings.Replace(string(compactImageHistory(`{"type":"input_image","image_url":"`+compactTestImageURL+`"}`)), `"compaction_trigger"`, `"function_call","call_id":"missing","name":"tool"`, 1))
	_, _, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), body, "deepseek-flash")
	require.Error(t, err)
	require.Empty(t, upstream.requests)
}

func TestDeepSeekCompactImageVersion2CapsuleRemainsReadable(t *testing.T) {
	svc, _ := compactImagesService()
	original := json.RawMessage(`{"type":"input_image","image_url":"` + compactTestImageURL + `"}`)
	legacyImages := []deepSeekCompactImage{{
		ID:          legacyCompactImageID(original),
		Original:    original,
		Description: "Legacy screenshot description.",
	}}
	state := deepSeekCompactEnvelope{Version: 2, Checkpoint: "legacy image checkpoint", Images: legacyImages}
	payload, err := marshalOpenAIUpstreamJSON(state)
	require.NoError(t, err)
	aead, err := svc.deepSeekCompactAEAD()
	require.NoError(t, err)
	nonce := make([]byte, aead.NonceSize())
	aad, err := deepSeekCompactUserAAD(deepSeekCompactTestContext(42))
	require.NoError(t, err)
	sealed := aead.Seal(nonce, nonce, payload, aad)
	envelope := deepSeekCompactEnvelopePrefix + base64.RawURLEncoding.EncodeToString(sealed)

	opened, err := svc.openDeepSeekCompactState(deepSeekCompactTestContext(42), envelope)
	require.NoError(t, err)
	require.Equal(t, 2, opened.Version)
	require.Equal(t, legacyImages, opened.Images)
	require.Contains(t, renderDeepSeekCompactState(opened), "Legacy screenshot description.")
}

func TestDeepSeekCompactImagesLargeInlineScreenshotUsesBoundedReference(t *testing.T) {
	svc, upstream := compactImagesService()
	upstream.resp = compactVisionResponse("completed", `[{"type":"output_text","text":"Large screenshot with a settings dialog."}]`)
	imageData := strings.Repeat("A", 400*1024)
	source := "https://images.example/screenshot.png?" + imageData
	body := compactImageHistory(`{"type":"input_image","image_url":"` + source + `","detail":"high"}`)

	textBody, images, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), body, "deepseek-flash")
	require.NoError(t, err)
	require.NotContains(t, string(textBody), imageData)
	require.Len(t, images, 1)
	require.NotEmpty(t, images[0].Original)
	require.Len(t, upstream.requests, 1)

	encrypted, err := svc.sealDeepSeekCompactState(deepSeekCompactTestContext(42), deepSeekCompactEnvelope{Checkpoint: "summary", Images: images})
	require.NoError(t, err)
	require.LessOrEqual(t, len(encrypted), deepSeekCompactMaxEnvelopeBytes)
	state, err := svc.openDeepSeekCompactState(deepSeekCompactTestContext(42), encrypted)
	require.NoError(t, err)
	require.Equal(t, deepSeekCompactImageVersion, state.Version)
	require.Len(t, state.Images, 1)
	require.Empty(t, state.Images[0].Original)
	require.Equal(t, images[0].ID, state.Images[0].ID)
	require.Equal(t, images[0].Description, state.Images[0].Description)

	second := mustMarshalDeepSeekCompactTestJSON(t, map[string]any{"model": "gpt-5.6-sol", "input": []any{map[string]any{"type": "compaction", "encrypted_content": encrypted}, map[string]any{"type": "compaction_trigger"}}})
	_, reused, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), second, "deepseek-flash")
	require.NoError(t, err)
	require.Equal(t, state.Images, reused)
	require.Len(t, upstream.requests, 1, "persisted descriptions must be reused")
}

func TestDeepSeekCompactImageVersion2CapsuleDoesNotDuplicateVisibleImage(t *testing.T) {
	svc, upstream := compactImagesService()
	original := json.RawMessage(`{"type":"input_image","image_url":"` + compactTestImageURL + `","detail":"high"}`)
	legacyImages := []deepSeekCompactImage{{
		ID:          legacyCompactImageID(original),
		Original:    original,
		Description: "Legacy screenshot description.",
	}}
	state := deepSeekCompactEnvelope{Version: 2, Checkpoint: "legacy image checkpoint", Images: legacyImages}
	payload, err := marshalOpenAIUpstreamJSON(state)
	require.NoError(t, err)
	aead, err := svc.deepSeekCompactAEAD()
	require.NoError(t, err)
	nonce := make([]byte, aead.NonceSize())
	aad, err := deepSeekCompactUserAAD(deepSeekCompactTestContext(42))
	require.NoError(t, err)
	sealed := aead.Seal(nonce, nonce, payload, aad)
	envelope := deepSeekCompactEnvelopePrefix + base64.RawURLEncoding.EncodeToString(sealed)

	body := mustMarshalDeepSeekCompactTestJSON(t, map[string]any{"model": "gpt-5.6-sol", "input": []any{
		map[string]any{"type": "compaction", "encrypted_content": envelope},
		map[string]any{"type": "message", "role": "user", "content": []any{
			map[string]any{"type": "input_image", "image_url": compactTestImageURL, "detail": "high"},
		}},
		map[string]any{"type": "compaction_trigger"},
	}})
	_, images, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), body, "deepseek-flash")
	require.NoError(t, err)
	require.Len(t, images, 1)
	require.Equal(t, "Legacy screenshot description.", images[0].Description)
	require.Empty(t, upstream.requests)
}

func TestDeepSeekCompactImagesLargeInlineBase64UsesBoundedReference(t *testing.T) {
	svc, upstream := compactImagesService()
	upstream.resp = compactVisionResponse("completed", `[{"type":"output_text","text":"Large inline screenshot."}]`)
	imageData := strings.Repeat("A", 400*1024)
	source := "data:image/png;base64," + imageData
	body := compactImageHistory(`{"type":"input_image","image_url":"` + source + `"}`)

	textBody, images, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), body, "deepseek-flash")
	require.NoError(t, err)
	require.NotContains(t, string(textBody), imageData)
	require.Len(t, images, 1)
	require.Len(t, upstream.requests, 1)
	require.Equal(t, source, gjson.GetBytes(upstream.bodies[0], "input.0.content.1.image_url").String())
	encrypted, err := svc.sealDeepSeekCompactState(deepSeekCompactTestContext(42), deepSeekCompactEnvelope{Checkpoint: "summary", Images: images})
	require.NoError(t, err)
	require.LessOrEqual(t, len(encrypted), deepSeekCompactMaxEnvelopeBytes)
	state, err := svc.openDeepSeekCompactState(deepSeekCompactTestContext(42), encrypted)
	require.NoError(t, err)
	require.Empty(t, state.Images[0].Original)
}

func TestDeepSeekCompactImagesDropsForeignCompaction(t *testing.T) {
	svc, upstream := compactImagesService()
	upstream.resp = compactVisionResponse("completed", `[{"type":"output_text","text":"Screenshot description."}]`)
	body := []byte(`{"model":"gpt-5.6-sol","input":[` +
		`{"type":"compaction","encrypted_content":"foreign-provider-checkpoint"},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"visible history"},{"type":"input_image","image_url":"` + compactTestImageURL + `"}]},` +
		`{"type":"compaction_trigger"}]}`)

	textBody, images, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), body, "deepseek-flash")
	require.NoError(t, err)
	require.Len(t, images, 1)
	require.NotContains(t, string(textBody), "foreign-provider-checkpoint")
	require.Contains(t, string(textBody), "visible history")
	require.Len(t, upstream.requests, 1)
	require.NotContains(t, string(upstream.bodies[0]), "foreign-provider-checkpoint")
}

func TestDeepSeekCompactImagesForeignCompactionNeedsVisibleHistory(t *testing.T) {
	svc, upstream := compactImagesService()
	body := []byte(`{"model":"gpt-5.6-sol","input":[{"type":"compaction","encrypted_content":"foreign-provider-checkpoint"},{"type":"compaction_trigger"}]}`)

	_, _, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), body, "deepseek-flash")
	require.ErrorContains(t, err, "no conversation history")
	require.Empty(t, upstream.requests)
}

func TestDeepSeekCompactImagesOwnedCompactionStillRequiresDecryption(t *testing.T) {
	svc, upstream := compactImagesService()
	body := []byte(`{"model":"gpt-5.6-sol","input":[` +
		`{"type":"compaction","encrypted_content":"` + deepSeekCompactEnvelopePrefix + `invalid"},` +
		`{"type":"message","role":"user","content":"visible history"},` +
		`{"type":"compaction_trigger"}]}`)

	_, _, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), body, "deepseek-flash")
	require.ErrorIs(t, err, ErrDeepSeekCompactInvalidEncryptedContent)
	require.Empty(t, upstream.requests)
}

func TestDeepSeekCompactImagesVisionFailures(t *testing.T) {
	for name, response := range map[string]*http.Response{
		"incomplete": compactVisionResponse("incomplete", `[{"type":"output_text","text":"partial"}]`),
		"empty":      compactVisionResponse("completed", `[{"type":"output_text","text":""}]`),
		"refusal":    compactVisionResponse("completed", `[{"type":"refusal","refusal":"cannot"}]`),
		"http_error": {StatusCode: 503, Body: io.NopCloser(strings.NewReader("Authorization: secret data:image/png"))},
	} {
		t.Run(name, func(t *testing.T) {
			svc, upstream := compactImagesService()
			upstream.resp = response
			_, _, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), compactImageHistory(`{"type":"input_image","image_url":"`+compactTestImageURL+`"}`), "deepseek-flash")
			require.Error(t, err)
			require.NotContains(t, err.Error(), "Authorization")
			require.Len(t, upstream.requests, 1)
		})
	}
	svc, _ := compactImagesService()
	svc.cfg.Gateway.DeepSeekCompactVisionAccountID = 0
	_, _, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), compactImageHistory(`{"type":"input_image","image_url":"`+compactTestImageURL+`"}`), "deepseek-flash")
	require.ErrorContains(t, err, "configure a vision account")
}

func TestDeepSeekCompactImagesHTTPAndWSCore(t *testing.T) {
	for _, withTrigger := range []bool{true, false} {
		t.Run(map[bool]string{true: "trigger", false: "compact_endpoint"}[withTrigger], func(t *testing.T) {
			svc, upstream := compactImagesService()
			upstream.responses = []*http.Response{compactVisionResponse("completed", `[{"type":"output_text","text":"Visible plugin menu labels."}]`), deepSeekRemoteCompactResponsesResponse("Summary with no image details.")}
			body := compactImageHistory(`{"type":"input_image","image_url":"` + compactTestImageURL + `"}`)
			if !withTrigger {
				body = []byte(strings.Replace(string(body), `,{"type":"compaction_trigger"}`, "", 1))
			}
			c, _ := deepSeekResponsesTestContext(t, body)
			result, err := svc.buildDeepSeekRemoteCompaction(deepSeekCompactTestContext(42), c, openaiMappedDeepSeekResponsesAccount(), body, "gpt-5.6-sol", "deepseek-flash", "deepseek-flash")
			require.NoError(t, err)
			require.Len(t, upstream.requests, 2)
			require.Equal(t, "Bearer sk-test", upstream.requests[1].Header.Get("Authorization"))
			require.NotContains(t, string(upstream.bodies[1]), compactTestImageURL)
			require.Contains(t, string(upstream.bodies[1]), "Visible plugin menu")
			state, err := svc.openDeepSeekCompactState(deepSeekCompactTestContext(42), gjson.GetBytes(result.Compaction, "encrypted_content").String())
			require.NoError(t, err)
			require.Len(t, state.Images, 1)
			require.Equal(t, "Summary with no image details.", strings.TrimPrefix(state.Checkpoint, deepSeekCompactCheckpointPreamble+"\n\n<compacted-summary>")[:len("Summary with no image details.")])
			events := [][]byte{}
			require.NoError(t, writeDeepSeekWSCompactionEvents(result.FinalJSON, func(data []byte) error { events = append(events, data); return nil }))
			require.Len(t, events, 2)
			require.Len(t, gjson.GetBytes(events[1], "response.output").Array(), 1)
		})
	}
}

func TestDeepSeekCompactImagesCancelledContext(t *testing.T) {
	svc, upstream := compactImagesService()
	upstream.err = context.Canceled
	ctx, cancel := context.WithCancel(deepSeekCompactTestContext(42))
	cancel()
	_, _, err := svc.prepareDeepSeekCompactImages(ctx, compactImageHistory(`{"type":"input_image","image_url":"`+compactTestImageURL+`"}`), "deepseek-flash")
	require.Error(t, err)
}

func TestDeepSeekCompactImagesNearLimitReuseAndAliases(t *testing.T) {
	for _, part := range []string{
		`{"type":"image_url","image_url":{"url":"` + compactTestImageURL + `"}}`,
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}}`,
		`{"type":"input_image","image_url":"https://images.example/` + strings.Repeat("a", 70000) + `"}`,
	} {
		svc, upstream := compactImagesService()
		upstream.resp = compactVisionResponse("completed", `[{"type":"output_text","text":"Retained description."}]`)
		_, images, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), compactImageHistory(part), "deepseek-flash")
		require.NoError(t, err)
		encrypted, err := svc.sealDeepSeekCompactState(deepSeekCompactTestContext(42), deepSeekCompactEnvelope{Checkpoint: "checkpoint", Images: images})
		require.NoError(t, err)
		second := mustMarshalDeepSeekCompactTestJSON(t, map[string]any{"model": "gpt-5.6-sol", "input": []any{map[string]any{"type": "compaction", "encrypted_content": encrypted}, map[string]any{"type": "compaction_trigger"}}})
		_, reused, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), second, "deepseek-flash")
		require.NoError(t, err)
		require.Equal(t, images, reused)
		require.Len(t, upstream.requests, 1)
	}
}

func TestDeepSeekCompactImagesStillEnforcePersistedCapsuleLimit(t *testing.T) {
	svc, _ := compactImagesService()
	images := make([]deepSeekCompactImage, compactImageMaxCount)
	for i := range images {
		images[i] = deepSeekCompactImage{
			ID:          compactImageDigest("image-" + strings.Repeat("x", i+1)),
			Description: strings.Repeat("d", compactImageDescriptionLimit),
		}
	}
	_, err := svc.sealDeepSeekCompactState(deepSeekCompactTestContext(42), deepSeekCompactEnvelope{
		Checkpoint: strings.Repeat("c", deepSeekCompactMaxSummaryBytes),
		Images:     images,
	})
	require.ErrorContains(t, err, "encrypted_content exceeds")
}

func TestDeepSeekCompactImagesForwardEntrypoints(t *testing.T) {
	for _, mode := range []string{"http", "ws"} {
		t.Run(mode, func(t *testing.T) {
			svc, upstream := compactImagesService()
			upstream.responses = []*http.Response{compactVisionResponse("completed", `[{"type":"output_text","text":"Token usage and automatic message history."}]`), deepSeekRemoteCompactResponsesResponse("checkpoint")}
			body := compactImageHistory(`{"type":"input_image","image_url":"` + compactTestImageURL + `"}`)
			c, recorder := deepSeekResponsesTestContext(t, body)
			if mode == "http" {
				c.Request.URL.Path = "/v1/responses/compact"
				result, err := svc.Forward(deepSeekCompactTestContext(42), c, openaiMappedDeepSeekResponsesAccount(), body)
				require.NoError(t, err)
				require.NotNil(t, result)
				require.Len(t, gjson.Get(recorder.Body.String(), "output").Array(), 1)
				require.Equal(t, "compaction", gjson.Get(recorder.Body.String(), "output.0.type").String())
			} else {
				var events [][]byte
				result, err := svc.proxyOpenAIWSHTTPBridgeTurn(deepSeekCompactTestContext(42), c, openaiMappedDeepSeekResponsesAccount(), "sk-test", body, len(body), "gpt-5.6-sol", "", "", "", "", 1, func(data []byte) error { events = append(events, data); return nil })
				require.NoError(t, err)
				require.NotEmpty(t, result.wsDeepSeekCompactionCheckpoint)
				require.Len(t, events, 2)
				require.Len(t, gjson.GetBytes(events[1], "response.output").Array(), 1)
			}
			require.Len(t, upstream.requests, 2)
		})
	}
}

func TestDeepSeekCompactImagesSelectedVisionRoute(t *testing.T) {
	for _, tc := range []struct {
		name      string
		base      string
		selected  string
		wantError bool
	}{
		{"mixed_account", "https://vision.example/v1", "gpt-6.1-sol", false},
		{"selected_deepseek", "https://vision.example/v1", "deepseek-flash", true},
		{"deepseek_host", "https://api.deepseek.com/v1", "gpt-6.1-sol", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc, upstream := compactImagesService()
			svc.accountRepo = stubOpenAIAccountRepo{accounts: []Account{{
				ID: 4, Platform: PlatformOpenAI, Type: AccountTypeAPIKey, Status: StatusActive, Schedulable: true,
				Credentials: map[string]any{"base_url": tc.base, "api_key": "vision-secret", "model_mapping": map[string]any{"gpt-5.6-sol": "deepseek-flash", "gpt-6.1-sol": tc.selected}},
			}}}
			upstream.resp = compactVisionResponse("completed", `[{"type":"output_text","text":"Screenshot labels."}]`)
			_, _, err := svc.prepareDeepSeekCompactImages(deepSeekCompactTestContext(42), compactImageHistory(`{"type":"input_image","image_url":"`+compactTestImageURL+`"}`), "deepseek-flash")
			if tc.wantError {
				require.ErrorContains(t, err, "vision route must not resolve")
				require.Empty(t, upstream.requests)
			} else {
				require.NoError(t, err)
				require.Len(t, upstream.requests, 1)
				require.Equal(t, "gpt-6.1-sol", gjson.GetBytes(upstream.bodies[0], "model").String())
			}
		})
	}
}
