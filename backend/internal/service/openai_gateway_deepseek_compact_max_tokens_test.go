package service

import (
	"encoding/json"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/tidwall/gjson"
)

func deepSeekCompactRequestFixture() []byte {
	return []byte(`{
		"model": "gpt-5.5",
		"input": [
			{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
			{"type": "compaction_trigger"}
		]
	}`)
}

func TestDeepSeekCompactResponsesRequestOmitsMaxOutputTokensByDefault(t *testing.T) {
	svc := &OpenAIGatewayService{cfg: &config.Config{}}

	body, err := svc.deepSeekCompactResponsesRequest(deepSeekCompactRequestFixture(), "deepseek-flash")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gjson.GetBytes(body, "max_output_tokens").Exists() {
		t.Fatalf("expected max_output_tokens to be omitted, got %s", body)
	}

	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
}

func TestDeepSeekCompactResponsesRequestAppliesConfiguredMaxOutputTokens(t *testing.T) {
	svc := &OpenAIGatewayService{cfg: &config.Config{Gateway: config.GatewayConfig{OpenAICompactSummaryMaxTokens: 32768}}}

	body, err := svc.deepSeekCompactResponsesRequest(deepSeekCompactRequestFixture(), "deepseek-flash")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := gjson.GetBytes(body, "max_output_tokens").Int(); got != 32768 {
		t.Fatalf("expected 32768, got %d", got)
	}
}

func TestDeepSeekCompactResponsesRequestRejectsNegativeMaxOutputTokens(t *testing.T) {
	svc := &OpenAIGatewayService{cfg: &config.Config{Gateway: config.GatewayConfig{OpenAICompactSummaryMaxTokens: -1}}}

	if _, err := svc.deepSeekCompactResponsesRequest(deepSeekCompactRequestFixture(), "deepseek-flash"); err == nil {
		t.Fatal("expected error for negative max_output_tokens")
	}
}

