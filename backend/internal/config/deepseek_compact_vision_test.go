package config

import (
	"github.com/stretchr/testify/require"
	"testing"
)

func TestDeepSeekCompactVisionConfiguration(t *testing.T) {
	resetViperWithJWTSecret(t)
	cfg, err := Load()
	require.NoError(t, err)
	require.Zero(t, cfg.Gateway.DeepSeekCompactVisionAccountID)
	require.Empty(t, cfg.Gateway.DeepSeekCompactVisionModel)
	t.Setenv("GATEWAY_DEEPSEEK_COMPACT_VISION_ACCOUNT_ID", "4")
	t.Setenv("GATEWAY_DEEPSEEK_COMPACT_VISION_MODEL", "gpt-6.1-sol")
	cfg, err = Load()
	require.NoError(t, err)
	require.Equal(t, int64(4), cfg.Gateway.DeepSeekCompactVisionAccountID)
	require.Equal(t, "gpt-6.1-sol", cfg.Gateway.DeepSeekCompactVisionModel)
}
