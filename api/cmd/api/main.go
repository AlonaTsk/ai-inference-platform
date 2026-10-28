// Command api serves the public prediction API in front of the inference service.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/AlonaTsk/ai-inference-platform/api/internal/server"
)

const shutdownTimeout = 10 * time.Second

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(log)

	if err := run(log); err != nil {
		log.Error("api failed", "err", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	timeout, err := time.ParseDuration(getenv("INFERENCE_TIMEOUT", "3s"))
	if err != nil {
		return fmt.Errorf("invalid INFERENCE_TIMEOUT: %w", err)
	}
	cfg := server.Config{
		InferenceURL: getenv("INFERENCE_URL", "http://localhost:8000"),
		Timeout:      timeout,
	}

	srv := &http.Server{
		Addr:              ":" + getenv("PORT", "8080"),
		Handler:           server.New(cfg, log).Routes(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      timeout + 5*time.Second, // must outlast the upstream call
		IdleTimeout:       60 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	serveErr := make(chan error, 1)
	go func() {
		log.Info("api listening", "addr", srv.Addr, "inference_url", cfg.InferenceURL)
		serveErr <- srv.ListenAndServe()
	}()

	select {
	case err := <-serveErr:
		return err // ListenAndServe never returns nil
	case <-ctx.Done():
	}

	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("shutdown: %w", err)
	}
	if err := <-serveErr; !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
