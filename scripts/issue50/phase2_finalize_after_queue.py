#!/usr/bin/env python3
"""Compatibility entry point for the Phase 2 evidence-only finalizer.

The queue is complete. This entry point deliberately cannot recover or start
training; it only rebuilds audited summaries from immutable artifacts.
"""

from phase2_audit_rebuild import main


if __name__ == "__main__":
    raise SystemExit(main())
