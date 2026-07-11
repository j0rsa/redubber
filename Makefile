.PHONY: help install install-backend install-frontend dev dev-backend dev-frontend story build test lint format clean docker-build docker-up docker-down

# Default target
help:
	@echo "Redubber v2.0 - Development Commands"
	@echo ""
	@echo "⚡ Quick Start:"
	@echo "  make quickstart       - Complete first-time setup (recommended)"
	@echo "  make setup            - Create directories and .env file"
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install all dependencies (backend + frontend)"
	@echo "  make install-backend  - Install Python dependencies via poetry"
	@echo "  make install-frontend - Install Node.js dependencies via npm"
	@echo ""
	@echo "Development:"
	@echo "  make dev              - Run backend + frontend in parallel"
	@echo "  make dev-backend      - Run FastAPI backend (port 8000)"
	@echo "  make dev-frontend     - Run Vite frontend (port 5173)"
	@echo "  make story            - Run Storybook component explorer (port 6006)"
	@echo ""
	@echo "Build:"
	@echo "  make build            - Build production frontend"
	@echo "  make docker-build     - Build Docker image"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run ruff linter"
	@echo "  make format           - Format code with ruff"
	@echo "  make typecheck        - Run pyright type checker"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up        - Start with docker-compose"
	@echo "  make docker-down      - Stop docker containers"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove build artifacts and cache"

# Installation
install: install-backend install-frontend
	@echo "✓ All dependencies installed"

install-backend:
	@echo "Installing Python dependencies..."
	poetry install
	@echo "✓ Backend dependencies installed"

install-frontend:
	@echo "Installing Node.js dependencies..."
	cd frontend && npm install
	@echo "✓ Frontend dependencies installed"

# Development
dev:
	@echo "Starting backend and frontend in parallel..."
	@$(MAKE) -j2 dev-backend dev-frontend

dev-backend:
	@echo "Starting FastAPI backend on http://localhost:8000"
	@echo "API docs available at http://localhost:8000/api/docs"
	poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app

dev-frontend:
	@echo "Starting Vite frontend on http://localhost:5173"
	cd frontend && npm run dev

story:
	@echo "Starting Storybook on http://localhost:6006"
	@echo "Explore 43 component stories with modern designs"
	cd frontend && npm run storybook

# Build
build:
	@echo "Building production frontend..."
	cd frontend && npm run build
	@echo "✓ Frontend built to frontend/dist/"

# Testing
test:
	@echo "Running all tests..."
	poetry run pytest -v

test-unit:
	@echo "Running unit tests..."
	poetry run pytest -v -m "not integration"

test-integration:
	@echo "Running integration tests..."
	poetry run pytest -v -m integration

# Code Quality
lint:
	@echo "Running ruff linter..."
	poetry run ruff check .

format:
	@echo "Formatting code with ruff..."
	poetry run ruff check --fix .
	poetry run ruff format .

typecheck:
	@echo "Running pyright type checker..."
	poetry run pyright

# Docker
docker-build:
	@echo "Building Docker image..."
	docker-compose build

docker-up:
	@echo "Starting containers with docker-compose..."
	docker-compose up -d
	@echo "✓ Redubber running at http://localhost:8000"

docker-down:
	@echo "Stopping containers..."
	docker-compose down

docker-logs:
	docker-compose logs -f

# Cleanup
clean:
	@echo "Cleaning build artifacts and cache..."
	rm -rf frontend/dist/
	rm -rf frontend/node_modules/.vite/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleanup complete"

# Database management
db-reset:
	@echo "Resetting database..."
	rm -f redubber.db
	@echo "✓ Database reset (will be recreated on next run)"

# Setup directories and environment
setup:
	@echo "Setting up local development environment..."
	@mkdir -p storage redubber_tmp
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env file from .env.example"; \
		echo "⚠️  IMPORTANT: Edit .env and set your OPENAI_API_KEY"; \
	else \
		echo "✓ .env file already exists"; \
	fi
	@echo "✓ Created storage directories"

# Quick start for first-time setup
quickstart: setup install
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "⚠️  IMPORTANT: Edit .env and set your OPENAI_API_KEY before running 'make dev'"
	@echo ""
	@echo "Then run: make dev"
	@echo ""
	@echo "Backend will run on: http://localhost:8000"
	@echo "Frontend will run on: http://localhost:5173"
	@echo "API docs available at: http://localhost:8000/api/docs"
