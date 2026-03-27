from __future__ import annotations

import unittest
from datetime import datetime, timezone

from mystripes.cookie_consent import (
    COOKIE_CONSENT_COOKIE_NAME,
    OPTIONAL_CONVENIENCE_PURPOSE,
    build_cookie_consent_payload,
    cookie_consent_choice,
    decode_cookie_consent_value,
    encode_cookie_consent_value,
    normalize_cookie_consent_choice,
    optional_cookie_consent_granted,
)


class CookieConsentTests(unittest.TestCase):
    def test_round_trip_for_accepted_choice(self) -> None:
        payload = build_cookie_consent_payload(
            "accepted",
            updated_at=datetime(2026, 3, 27, 15, 0, tzinfo=timezone.utc),
        )

        encoded = encode_cookie_consent_value(payload)
        decoded = decode_cookie_consent_value(encoded)

        self.assertEqual(decoded["choice"], "accepted")
        self.assertEqual(decoded["updated_at"], "2026-03-27T15:00:00Z")
        self.assertTrue(decoded["purposes"][OPTIONAL_CONVENIENCE_PURPOSE])
        self.assertTrue(optional_cookie_consent_granted(decoded))

    def test_round_trip_for_rejected_choice(self) -> None:
        encoded = encode_cookie_consent_value(
            build_cookie_consent_payload(
                "rejected",
                updated_at=datetime(2026, 3, 27, 15, 5, tzinfo=timezone.utc),
            )
        )
        decoded = decode_cookie_consent_value(encoded)

        self.assertEqual(cookie_consent_choice(decoded), "rejected")
        self.assertFalse(optional_cookie_consent_granted(decoded))

    def test_invalid_choice_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_cookie_consent_choice("maybe")

    def test_invalid_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            decode_cookie_consent_value("not-a-valid-consent-cookie")


if __name__ == "__main__":
    unittest.main()
