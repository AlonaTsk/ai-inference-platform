package server

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func newTestServer(t *testing.T, upstream http.HandlerFunc, timeout time.Duration) http.Handler {
	t.Helper()
	up := httptest.NewServer(upstream)
	t.Cleanup(up.Close)
	return New(Config{InferenceURL: up.URL, Timeout: timeout}, slog.New(slog.NewTextHandler(io.Discard, nil))).Routes()
}

func okUpstream(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/readyz":
			w.WriteHeader(http.StatusOK)
		case "/predict":
			var req upstreamRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Texts) != 1 {
				t.Errorf("bad upstream request: %v %v", err, req)
			}
			io.WriteString(w, `{"predictions":[{"label":"billing","confidence":0.91,
				"scores":{"billing":0.91,"technical":0.09}}],"model_version":"v1"}`)
		}
	}
}

func do(h http.Handler, method, path, body string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(method, path, strings.NewReader(body)))
	return rec
}

func TestPredictSuccess(t *testing.T) {
	t.Parallel()
	h := newTestServer(t, okUpstream(t), time.Second)
	rec := do(h, http.MethodPost, "/v1/predict", `{"text":"I was charged twice"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body)
	}
	var got PredictResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Label != "billing" || got.Confidence != 0.91 || got.ModelVersion != "v1" {
		t.Errorf("unexpected response: %+v", got)
	}
}

func TestPredictValidation(t *testing.T) {
	t.Parallel()
	h := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("upstream must not be called for invalid input")
	}, time.Second)

	cases := map[string]string{
		"not json":      `nope`,
		"empty text":    `{"text":"   "}`,
		"missing text":  `{}`,
		"unknown field": `{"text":"hi","extra":1}`,
		"too long":      `{"text":"` + strings.Repeat("a", MaxTextLen+1) + `"}`,
		"two objects":   `{"text":"a"}{"text":"b"}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if rec := do(h, http.MethodPost, "/v1/predict", body); rec.Code != http.StatusBadRequest {
				t.Errorf("status = %d, want 400", rec.Code)
			}
		})
	}
}

func TestPredictOversizedBodyIs413(t *testing.T) {
	t.Parallel()
	h := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("upstream must not be called for an oversized body")
	}, time.Second)
	body := `{"text":"` + strings.Repeat("a", maxBodyBytes) + `"}`
	if rec := do(h, http.MethodPost, "/v1/predict", body); rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("status = %d, want 413", rec.Code)
	}
}

func TestPredictUpstreamErrorIs502(t *testing.T) {
	t.Parallel()
	h := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}, time.Second)
	if rec := do(h, http.MethodPost, "/v1/predict", `{"text":"hi"}`); rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", rec.Code)
	}
}

func TestPredictMalformedUpstreamIs502(t *testing.T) {
	t.Parallel()
	h := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"predictions":[]}`)
	}, time.Second)
	if rec := do(h, http.MethodPost, "/v1/predict", `{"text":"hi"}`); rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", rec.Code)
	}
}

func TestPredictUpstreamTimeoutIs504(t *testing.T) {
	t.Parallel()
	h := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
	}, 50*time.Millisecond)
	if rec := do(h, http.MethodPost, "/v1/predict", `{"text":"hi"}`); rec.Code != http.StatusGatewayTimeout {
		t.Errorf("status = %d, want 504", rec.Code)
	}
}

func TestPredictUpstreamDownIs502(t *testing.T) {
	t.Parallel()
	h := New(Config{InferenceURL: "http://127.0.0.1:1", Timeout: time.Second},
		slog.New(slog.NewTextHandler(io.Discard, nil))).Routes()
	if rec := do(h, http.MethodPost, "/v1/predict", `{"text":"hi"}`); rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", rec.Code)
	}
}

func TestReadyzFollowsInference(t *testing.T) {
	t.Parallel()
	ready := newTestServer(t, okUpstream(t), time.Second)
	if rec := do(ready, http.MethodGet, "/readyz", ""); rec.Code != http.StatusOK {
		t.Errorf("ready: status = %d", rec.Code)
	}
	notReady := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}, time.Second)
	if rec := do(notReady, http.MethodGet, "/readyz", ""); rec.Code != http.StatusServiceUnavailable {
		t.Errorf("not ready: status = %d", rec.Code)
	}
}

func TestHealthzAndMethodRouting(t *testing.T) {
	t.Parallel()
	h := newTestServer(t, okUpstream(t), time.Second)
	if rec := do(h, http.MethodGet, "/healthz", ""); rec.Code != http.StatusOK {
		t.Errorf("healthz status = %d", rec.Code)
	}
	if rec := do(h, http.MethodGet, "/v1/predict", ""); rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET /v1/predict status = %d, want 405", rec.Code)
	}
}
