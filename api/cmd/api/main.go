package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/AlonaTsk/ai-inference-platform/api/internal/server"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	timeout, err := time.ParseDuration(getenv("INFERENCE_TIMEOUT", "3s"))
	if err != nil {
		log.Error("invalid INFERENCE_TIMEOUT", "err", err)
		os.Exit(1)
	}
	cfg := server.Config{
		InferenceURL: getenv("INFERENCE_URL", "http://localhost:8000"),
		Timeout:      timeout,
	}
	addr := ":" + getenv("PORT", "8080")

	srv := &http.Server{
		Addr:              addr,
		Handler:           server.New(cfg, log).Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Info("api listening", "addr", addr, "inference_url", cfg.InferenceURL)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("server failed", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error("shutdown failed", "err", err)
	}
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
