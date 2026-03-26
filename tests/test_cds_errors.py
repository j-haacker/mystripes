from __future__ import annotations

import unittest

from mystripes.cds import _explain_cds_error


class CDSErrorTests(unittest.TestCase):
    def test_explain_licence_error(self) -> None:
        message = _explain_cds_error(
            RuntimeError(
                "403 Client Error: Forbidden required licences not accepted"
            )
        )
        self.assertIn("required CDS licence has not been accepted", message)
        self.assertIn("manage-licences", message)

    def test_explain_unauthorized_error_mentions_bare_token(self) -> None:
        message = _explain_cds_error(
            RuntimeError("401 Client Error: Unauthorized")
        )
        self.assertIn("bare `CDSAPI_KEY` value", message)
        self.assertIn("not `user:token`", message)


if __name__ == "__main__":
    unittest.main()
