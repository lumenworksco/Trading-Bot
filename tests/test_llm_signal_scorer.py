"""Tests for llm_signal_scorer.py — LLM-based signal quality scoring."""

import json
from unittest.mock import patch, MagicMock

import pytest

from conftest import ET


def _make_signal():
    """Create a dummy Signal for testing."""
    from strategies.base import Signal
    return Signal(
        symbol="AAPL",
        strategy="ORB",
        side="buy",
        entry_price=180.0,
        take_profit=185.0,
        stop_loss=178.0,
        pair_id="",
        hold_type="day",
        reason="breakout above range",
    )


def _make_context():
    return {
        'spy_day_return': 0.005,
        'sector_day_return': 0.008,
        'vix_level': 15.2,
        'signal_z_score': 1.8,
        'recent_news_headlines': ['AAPL beats earnings expectations'],
        'recent_trades_symbol': ['buy @ 179.50 +0.3%'],
    }


class TestBudgetGuard:
    def test_budget_guard_returns_default(self, override_config):
        """When daily budget is exhausted, return default score without API call."""
        with override_config(
            LLM_SCORING_ENABLED=True,
            LLM_MAX_DAILY_COST_USD=0.10,
            ANTHROPIC_API_KEY='test-key',
        ):
            from llm_signal_scorer import LLMSignalScorer, SignalScore

            with patch('llm_signal_scorer.anthropic.Anthropic'):
                scorer = LLMSignalScorer()
                scorer._daily_cost_usd = 0.10  # At budget limit

                result = scorer.score_signal(_make_signal(), _make_context())

                assert result.score == 0.7
                assert result.confidence == 'LOW'
                assert result.reasoning == 'budget_exceeded'
                assert result.size_mult == 1.0
                # Client should NOT have been called
                scorer.client.messages.create.assert_not_called()


class TestSuccessfulScoring:
    def test_successful_scoring(self, override_config):
        """Successful API call returns parsed SignalScore."""
        with override_config(
            LLM_SCORING_ENABLED=True,
            LLM_MAX_DAILY_COST_USD=0.10,
            ANTHROPIC_API_KEY='test-key',
        ):
            from llm_signal_scorer import LLMSignalScorer, SignalScore

            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = json.dumps({
                'score': 0.85,
                'confidence': 'HIGH',
                'reasoning': 'Strong breakout with trend alignment',
                'size_mult': 1.3,
            })
            mock_response.usage.input_tokens = 500
            mock_response.usage.output_tokens = 50

            with patch('llm_signal_scorer.anthropic.Anthropic') as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.messages.create.return_value = mock_response

                scorer = LLMSignalScorer()
                result = scorer.score_signal(_make_signal(), _make_context())

                assert result.score == 0.85
                assert result.confidence == 'HIGH'
                assert result.reasoning == 'Strong breakout with trend alignment'
                assert result.size_mult == 1.3
                assert scorer._call_count == 1
                assert scorer._daily_cost_usd > 0


class TestAPIFailure:
    def test_api_failure_returns_default(self, override_config):
        """API exception returns fail-open default score."""
        with override_config(
            LLM_SCORING_ENABLED=True,
            LLM_MAX_DAILY_COST_USD=0.10,
            ANTHROPIC_API_KEY='test-key',
        ):
            from llm_signal_scorer import LLMSignalScorer, SignalScore

            with patch('llm_signal_scorer.anthropic.Anthropic') as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.messages.create.side_effect = Exception("API timeout")

                scorer = LLMSignalScorer()
                result = scorer.score_signal(_make_signal(), _make_context())

                assert result.score == 0.7
                assert result.confidence == 'LOW'
                assert result.reasoning == 'llm_error'
                assert result.size_mult == 1.0


class TestParseError:
    def test_parse_error_returns_default(self, override_config):
        """Malformed JSON response returns fail-open default score."""
        with override_config(
            LLM_SCORING_ENABLED=True,
            LLM_MAX_DAILY_COST_USD=0.10,
            ANTHROPIC_API_KEY='test-key',
        ):
            from llm_signal_scorer import LLMSignalScorer, SignalScore

            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "This is not valid JSON at all"
            mock_response.usage.input_tokens = 500
            mock_response.usage.output_tokens = 50

            with patch('llm_signal_scorer.anthropic.Anthropic') as MockClient:
                mock_instance = MockClient.return_value
                mock_instance.messages.create.return_value = mock_response

                scorer = LLMSignalScorer()
                result = scorer.score_signal(_make_signal(), _make_context())

                assert result.score == 0.7
                assert result.confidence == 'LOW'
                assert result.reasoning == 'llm_error'
                assert result.size_mult == 1.0


class TestResetDaily:
    def test_reset_daily(self, override_config):
        """reset_daily() zeroes out cost and call count."""
        with override_config(
            LLM_SCORING_ENABLED=True,
            LLM_MAX_DAILY_COST_USD=0.10,
            ANTHROPIC_API_KEY='test-key',
        ):
            from llm_signal_scorer import LLMSignalScorer

            with patch('llm_signal_scorer.anthropic.Anthropic'):
                scorer = LLMSignalScorer()
                scorer._call_count = 42
                scorer._daily_cost_usd = 0.08

                scorer.reset_daily()

                assert scorer._call_count == 0
                assert scorer._daily_cost_usd == 0.0
