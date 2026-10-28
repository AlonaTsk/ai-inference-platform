#!/usr/bin/env bash
# End-to-end smoke test against a running stack (Compose or a kubectl port-forward).
# Usage: tests/e2e.sh [base_url]   (default http://localhost:8080)
set -euo pipefail

BASE="${1:-http://localhost:8080}"
fail() { echo "FAIL: $*" >&2; exit 1; }

status() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

echo "waiting for $BASE/readyz"
for _ in $(seq 1 60); do
  [ "$(status "$BASE/readyz")" = 200 ] && break
  sleep 1
done
[ "$(status "$BASE/readyz")" = 200 ] || fail "API never became ready"

check_label() {
  local text="$1" want="$2" body got
  body=$(curl -sf -X POST "$BASE/v1/predict" -H 'Content-Type: application/json' -d "{\"text\": \"$text\"}") \
    || fail "predict request failed for: $text"
  got=$(echo "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["label"])')
  [ "$got" = "$want" ] || fail "'$text' -> $got, want $want ($body)"
  echo "ok   '$text' -> $got"
}

check_label "I was charged twice this month" billing
check_label "my package has not arrived yet" shipping
check_label "the password reset email never arrives" account
check_label "the app crashes when I open settings" technical

code=$(status -X POST "$BASE/v1/predict" -H 'Content-Type: application/json' -d '{"text": ""}')
[ "$code" = 400 ] || fail "empty text returned $code, want 400"
echo "ok   empty text -> 400"

echo "PASS"
