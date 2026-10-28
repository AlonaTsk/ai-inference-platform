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
	MaxTextLen   = 2000
	maxBodyBytes = 16 << 10
)

type Config struct {
	InferenceURL string
	Timeout      time.Duration
}

type Server struct {
	cfg    Config
	client *http.Client
	log    *slog.Logger
}

func New(cfg Config, log *slog.Logger) *Server {
	return &Server{
		cfg:    cfg,
		client: &http.Client{Timeout: cfg.Timeout},
		log:    log,
	}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.healthz)
	mux.HandleFunc("GET /readyz", s.readyz)
	mux.HandleFunc("POST /v1/predict", s.predict)
	return s.logRequests(mux)
}

type PredictRequest struct {
	Text string `json:"text"`
}

type PredictResponse struct {
	Label        string             `json:"label"`
	Confidence   float64            `json:"confidence"`
	Scores       map[string]float64 `json:"scores"`
	ModelVersion string             `json:"model_version,omitempty"`
}

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
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, s.cfg.InferenceURL+"/readyz", nil)
	resp, err := s.client.Do(req)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{"inference unreachable"})
		return
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{"inference not ready"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (s *Server) predict(w http.ResponseWriter, r *http.Request) {
	var in PredictRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{"body must be JSON: {\"text\": \"...\"}"})
		return
	}
	if err := validate(in.Text); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{err.Error()})
		return
	}

	out, status, err := s.callInference(r.Context(), in.Text)
	if err != nil {
		s.log.Error("inference call failed", "err", err, "status", status)
		writeJSON(w, status, ErrorResponse{http.StatusText(status)})
		return
	}
	writeJSON(w, http.StatusOK, out)
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
	body, _ := json.Marshal(upstreamRequest{Texts: []string{text}})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.InferenceURL+"/predict", bytes.NewReader(body))
	if err != nil {
		return nil, http.StatusInternalServerError, err
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

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, http.StatusBadGateway, fmt.Errorf("upstream status %d: %s", resp.StatusCode, snippet)
	}
	var up upstreamResponse
	if err := json.NewDecoder(resp.Body).Decode(&up); err != nil {
		return nil, http.StatusBadGateway, fmt.Errorf("decode upstream: %w", err)
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
	_ = json.NewEncoder(w).Encode(v)
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
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
