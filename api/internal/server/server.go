// Package server implements the public prediction API. It validates requests,
// forwards them to the Python inference service and maps upstream failures to
// stable, documented error responses.
package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	// MaxTextLen is the maximum ticket length in characters (runes).
	MaxTextLen = 2000

	maxBodyBytes     = 16 << 10
	maxUpstreamBytes = 64 << 10
	readyzTimeout    = 2 * time.Second
)

// Config holds the settings for talking to the inference service.
type Config struct {
	// InferenceURL is the base URL of the inference service, without a trailing slash.
	InferenceURL string
	// Timeout bounds each upstream request, including reading the response.
	Timeout time.Duration
}

// Server serves the prediction API. Create it with New.
type Server struct {
	cfg    Config
	client *http.Client
	log    *slog.Logger
}

// New returns a Server that forwards predictions to cfg.InferenceURL.
func New(cfg Config, log *slog.Logger) *Server {
	// The default transport keeps only 2 idle connections per host, which
	// forces new upstream connections under concurrent load.
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = 100
	transport.MaxIdleConnsPerHost = 100

	return &Server{
		cfg:    cfg,
		client: &http.Client{Timeout: cfg.Timeout, Transport: transport},
		log:    log,
	}
}

// Routes returns the HTTP handler with all endpoints and request logging.
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.healthz)
	mux.HandleFunc("GET /readyz", s.readyz)
	mux.HandleFunc("POST /v1/predict", s.predict)
	return s.logRequests(mux)
}

// PredictRequest is the body of POST /v1/predict.
type PredictRequest struct {
	Text string `json:"text"`
}

// PredictResponse is the successful response of POST /v1/predict.
type PredictResponse struct {
	Label        string             `json:"label"`
	Confidence   float64            `json:"confidence"`
	Scores       map[string]float64 `json:"scores"`
	ModelVersion string             `json:"model_version,omitempty"`
}

// ErrorResponse is the body of every non-2xx response.
type ErrorResponse struct {
	Error string `json:"error"`
}

type upstreamRequest struct {
	Texts []string `json:"texts"`
}

type upstreamResponse struct {
	Predictions []struct {
		Label      string             `json:"label"`
		Confidence float64            `json:"confidence"`
		Scores     map[string]float64 `json:"scores"`
	} `json:"predictions"`
	ModelVersion string `json:"model_version"`
}

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// readyz reports ready only when the inference service is ready, so a
// Kubernetes Service never routes traffic to an API pod that cannot answer.
func (s *Server) readyz(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), readyzTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.cfg.InferenceURL+"/readyz", nil)
	if err != nil {
		s.log.Error("build readyz request", "err", err)
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{"invalid inference URL"})
		return
	}
	resp, err := s.client.Do(req)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{"inference unreachable"})
		return
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxUpstreamBytes)) // allow connection reuse

	if resp.StatusCode != http.StatusOK {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{"inference not ready"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (s *Server) predict(w http.ResponseWriter, r *http.Request) {
	in, status, err := decodePredictRequest(w, r)
	if err != nil {
		writeJSON(w, status, ErrorResponse{err.Error()})
		return
	}

	out, status, err := s.callInference(r.Context(), in.Text)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			// The client went away; nobody is waiting for an answer.
			s.log.Info("client canceled request")
			return
		}
		s.log.Error("inference call failed", "err", err, "status", status)
		writeJSON(w, status, ErrorResponse{http.StatusText(status)})
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// decodePredictRequest parses and validates the body, returning the HTTP
// status to use when it is rejected.
func decodePredictRequest(w http.ResponseWriter, r *http.Request) (PredictRequest, int, error) {
	var in PredictRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	dec.DisallowUnknownFields()

	if err := dec.Decode(&in); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return in, http.StatusRequestEntityTooLarge, fmt.Errorf("body must be at most %d bytes", maxBodyBytes)
		}
		return in, http.StatusBadRequest, errors.New(`body must be JSON: {"text": "..."}`)
	}
	if dec.More() {
		return in, http.StatusBadRequest, errors.New("body must contain a single JSON object")
	}
	if err := validate(in.Text); err != nil {
		return in, http.StatusBadRequest, err
	}
	return in, 0, nil
}

func validate(text string) error {
	switch {
	case strings.TrimSpace(text) == "":
		return errors.New("text must not be empty")
	case !utf8.ValidString(text):
		return errors.New("text must be valid UTF-8")
	case utf8.RuneCountInString(text) > MaxTextLen:
		return fmt.Errorf("text must be at most %d characters", MaxTextLen)
	}
	return nil
}

// callInference returns the HTTP status to send the client on failure:
// 504 when the upstream timed out, 502 for any other upstream problem.
func (s *Server) callInference(ctx context.Context, text string) (*PredictResponse, int, error) {
	body, err := json.Marshal(upstreamRequest{Texts: []string{text}})
	if err != nil {
		return nil, http.StatusInternalServerError, fmt.Errorf("encode upstream request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.InferenceURL+"/predict", bytes.NewReader(body))
	if err != nil {
		return nil, http.StatusInternalServerError, fmt.Errorf("build upstream request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || isTimeout(err) {
			return nil, http.StatusGatewayTimeout, err
		}
		return nil, http.StatusBadGateway, err
	}
	defer resp.Body.Close()
	limited := io.LimitReader(resp.Body, maxUpstreamBytes)

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(limited, 512))
		return nil, http.StatusBadGateway, fmt.Errorf("upstream status %d: %s", resp.StatusCode, snippet)
	}
	var up upstreamResponse
	if err := json.NewDecoder(limited).Decode(&up); err != nil {
		return nil, http.StatusBadGateway, fmt.Errorf("decode upstream response: %w", err)
	}
	if len(up.Predictions) != 1 {
		return nil, http.StatusBadGateway, fmt.Errorf("expected 1 prediction, got %d", len(up.Predictions))
	}
	p := up.Predictions[0]
	return &PredictResponse{Label: p.Label, Confidence: p.Confidence, Scores: p.Scores, ModelVersion: up.ModelVersion}, 0, nil
}

func isTimeout(err error) bool {
	var t interface{ Timeout() bool }
	return errors.As(err, &t) && t.Timeout()
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v) // the status is already sent; nothing useful to do on error
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// Unwrap lets http.ResponseController reach the underlying writer.
func (r *statusRecorder) Unwrap() http.ResponseWriter {
	return r.ResponseWriter
}

func (s *Server) logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		if r.URL.Path == "/healthz" || r.URL.Path == "/readyz" {
			return
		}
		s.log.Info("request", "method", r.Method, "path", r.URL.Path, "status", rec.status,
			"duration_ms", time.Since(start).Milliseconds())
	})
}
