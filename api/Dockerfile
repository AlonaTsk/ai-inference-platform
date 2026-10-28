FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/api ./cmd/api

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/api /api
EXPOSE 8080
# Numeric UID (distroless "nonroot" = 65532) so Kubernetes runAsNonRoot can verify it.
USER 65532:65532
ENTRYPOINT ["/api"]
