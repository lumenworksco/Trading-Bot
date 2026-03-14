"""Tests for news_sentiment.py — Alpaca News sentiment filter."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch


class TestAlpacaNewsSentiment(unittest.TestCase):
    """Unit tests for AlpacaNewsSentiment."""

    def _make_sentiment(self, mock_client_cls):
        """Construct an AlpacaNewsSentiment with a mocked NewsClient."""
        mock_client_cls.return_value = MagicMock()
        from news_sentiment import AlpacaNewsSentiment
        obj = AlpacaNewsSentiment()
        return obj, mock_client_cls.return_value

    @patch("alpaca.data.historical.news.NewsClient", autospec=True)
    def test_no_news_returns_neutral(self, mock_client_cls):
        """No articles -> (1.0, 'no_news')."""
        from news_sentiment import AlpacaNewsSentiment

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # get_news returns object with empty news list
        mock_response = MagicMock()
        mock_response.news = []
        mock_client.get_news.return_value = mock_response

        sentiment = AlpacaNewsSentiment()
        mult, reason = sentiment.get_sentiment_size_mult("AAPL")

        self.assertEqual(mult, 1.0)
        self.assertEqual(reason, "no_news")

    @patch("alpaca.data.historical.news.NewsClient", autospec=True)
    def test_halt_keyword_detected(self, mock_client_cls):
        """Headline containing a halt keyword -> (0.0, ...)."""
        from news_sentiment import AlpacaNewsSentiment

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        article = MagicMock()
        article.headline = "Company files for bankruptcy protection amid debt crisis"
        mock_response = MagicMock()
        mock_response.news = [article]
        mock_client.get_news.return_value = mock_response

        sentiment = AlpacaNewsSentiment()
        mult, reason = sentiment.get_sentiment_size_mult("XYZ")

        self.assertEqual(mult, 0.0)
        self.assertIn("halt_keyword", reason)
        self.assertIn("bankruptcy", reason)

    @patch("alpaca.data.historical.news.NewsClient", autospec=True)
    def test_negative_sentiment(self, mock_client_cls):
        """Two negative headlines -> score <= -3 -> mult 0.5."""
        from news_sentiment import AlpacaNewsSentiment

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        a1 = MagicMock()
        a1.headline = "Company misses earnings estimates badly"
        a2 = MagicMock()
        a2.headline = "Analyst downgrade: stock cut to sell"
        mock_response = MagicMock()
        mock_response.news = [a1, a2]
        mock_client.get_news.return_value = mock_response

        sentiment = AlpacaNewsSentiment()
        mult, reason = sentiment.get_sentiment_size_mult("BAD")

        # misses = -2, downgrade = -2 => net -4 => very_negative => 0.5
        self.assertEqual(mult, 0.5)
        self.assertIn("very_negative", reason)

    @patch("alpaca.data.historical.news.NewsClient", autospec=True)
    def test_positive_sentiment(self, mock_client_cls):
        """Headlines with strong positive keywords -> mult 1.1."""
        from news_sentiment import AlpacaNewsSentiment

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        a1 = MagicMock()
        a1.headline = "Company beats Q3 estimates, raises guidance for FY"
        mock_response = MagicMock()
        mock_response.news = [a1]
        mock_client.get_news.return_value = mock_response

        sentiment = AlpacaNewsSentiment()
        mult, reason = sentiment.get_sentiment_size_mult("GOOD")

        # beats = +2, raises guidance = +2 => net +4 => positive => 1.1
        self.assertEqual(mult, 1.1)
        self.assertIn("positive", reason)

    @patch("alpaca.data.historical.news.NewsClient", autospec=True)
    def test_cache_hit(self, mock_client_cls):
        """Second call for same symbol within TTL should not call API again."""
        from news_sentiment import AlpacaNewsSentiment

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.news = []
        mock_client.get_news.return_value = mock_response

        sentiment = AlpacaNewsSentiment()

        # First call — hits API
        sentiment.get_sentiment_size_mult("AAPL")
        self.assertEqual(mock_client.get_news.call_count, 1)

        # Second call — should use cache
        mult, reason = sentiment.get_sentiment_size_mult("AAPL")
        self.assertEqual(mock_client.get_news.call_count, 1)  # still 1
        self.assertEqual(mult, 1.0)

    @patch("alpaca.data.historical.news.NewsClient", autospec=True)
    def test_api_failure_returns_default(self, mock_client_cls):
        """If NewsClient raises, return (1.0, 'news_unavailable')."""
        from news_sentiment import AlpacaNewsSentiment

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_news.side_effect = Exception("API timeout")

        sentiment = AlpacaNewsSentiment()
        mult, reason = sentiment.get_sentiment_size_mult("FAIL")

        self.assertEqual(mult, 1.0)
        self.assertEqual(reason, "news_unavailable")


if __name__ == "__main__":
    unittest.main()
