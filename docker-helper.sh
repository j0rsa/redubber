#!/usr/bin/env bash
# Docker Helper Script for Redubber v2.0
# Provides convenient commands for common Docker operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Redubber v2.0 - Docker Helper${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo ""
}

show_usage() {
    print_header
    echo "Usage: ./docker-helper.sh <command>"
    echo ""
    echo "Commands:"
    echo "  build           Build the Docker image"
    echo "  start           Start the service (detached)"
    echo "  stop            Stop the service"
    echo "  restart         Restart the service"
    echo "  logs            Follow service logs"
    echo "  status          Show service status"
    echo "  health          Check application health"
    echo "  shell           Open a shell in the container"
    echo "  clean           Stop service and clean volumes (WARNING: deletes data)"
    echo "  rebuild         Rebuild image from scratch (no cache)"
    echo "  backup          Backup database and storage"
    echo "  restore <file>  Restore from backup file"
    echo ""
}

# Check if .env exists
check_env() {
    if [ ! -f .env ]; then
        print_warning ".env file not found!"
        print_info "Creating .env from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env and set your OPENAI_API_KEY before starting the service"
        exit 1
    fi
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker daemon is not running"
        print_info "Please start Docker and try again"
        exit 1
    fi
}

# Build command
cmd_build() {
    print_header
    print_info "Building Docker image..."
    docker-compose build
    print_success "Build completed successfully"
}

# Start command
cmd_start() {
    print_header
    check_env
    print_info "Starting Redubber service..."
    docker-compose up -d
    print_success "Service started"
    print_info "Access the application at: http://localhost:8000"
    print_info "API documentation at: http://localhost:8000/api/docs"
    print_info "View logs with: ./docker-helper.sh logs"
}

# Stop command
cmd_stop() {
    print_header
    print_info "Stopping Redubber service..."
    docker-compose down
    print_success "Service stopped"
}

# Restart command
cmd_restart() {
    print_header
    print_info "Restarting Redubber service..."
    docker-compose restart
    print_success "Service restarted"
}

# Logs command
cmd_logs() {
    print_header
    print_info "Following service logs (Ctrl+C to exit)..."
    docker-compose logs -f redubber
}

# Status command
cmd_status() {
    print_header
    print_info "Service Status:"
    docker-compose ps
    echo ""

    if docker ps | grep -q redubber-v2; then
        print_info "Health Status:"
        docker inspect redubber-v2 --format='  Status: {{.State.Health.Status}}'

        print_info "Resource Usage:"
        docker stats redubber-v2 --no-stream --format "  CPU: {{.CPUPerc}}\n  Memory: {{.MemUsage}}"
    fi
}

# Health command
cmd_health() {
    print_header
    print_info "Checking application health..."

    if ! docker ps | grep -q redubber-v2; then
        print_error "Container is not running"
        exit 1
    fi

    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/health)

    if [ "$response" = "200" ]; then
        print_success "Application is healthy"
        curl -s http://localhost:8000/api/health | python3 -m json.tool || true
    else
        print_error "Application health check failed (HTTP $response)"
        exit 1
    fi
}

# Shell command
cmd_shell() {
    print_header
    print_info "Opening shell in container..."
    docker-compose exec redubber /bin/bash || docker-compose exec redubber /bin/sh
}

# Clean command
cmd_clean() {
    print_header
    print_warning "This will stop the service and remove all volumes (database and storage)"
    read -p "Are you sure? Type 'yes' to confirm: " -r
    echo

    if [[ ! $REPLY == "yes" ]]; then
        print_info "Aborted"
        exit 0
    fi

    print_info "Stopping service and cleaning volumes..."
    docker-compose down -v

    if [ -d storage ]; then
        print_info "Removing storage directory..."
        rm -rf storage/*
    fi

    if [ -d tmp ]; then
        print_info "Removing tmp directory..."
        rm -rf tmp/*
    fi

    print_success "Cleanup completed"
}

# Rebuild command
cmd_rebuild() {
    print_header
    print_info "Rebuilding image from scratch (no cache)..."
    docker-compose build --no-cache
    print_success "Rebuild completed successfully"
}

# Backup command
cmd_backup() {
    print_header

    if [ ! -d storage ]; then
        print_error "storage/ directory not found"
        exit 1
    fi

    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_file="redubber-backup-${timestamp}.tar.gz"

    print_info "Creating backup: $backup_file"

    # Stop container for consistent backup (optional)
    read -p "Stop container for consistent backup? [y/N]: " -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose stop redubber
        tar -czf "$backup_file" storage/
        docker-compose start redubber
    else
        tar -czf "$backup_file" storage/
    fi

    print_success "Backup created: $backup_file"
    print_info "Backup size: $(du -h "$backup_file" | cut -f1)"
}

# Restore command
cmd_restore() {
    print_header

    if [ -z "$1" ]; then
        print_error "Please specify backup file to restore"
        echo "Usage: ./docker-helper.sh restore <backup-file>"
        exit 1
    fi

    backup_file="$1"

    if [ ! -f "$backup_file" ]; then
        print_error "Backup file not found: $backup_file"
        exit 1
    fi

    print_warning "This will overwrite current storage data"
    read -p "Are you sure? Type 'yes' to confirm: " -r
    echo

    if [[ ! $REPLY == "yes" ]]; then
        print_info "Aborted"
        exit 0
    fi

    print_info "Stopping service..."
    docker-compose down

    print_info "Restoring from: $backup_file"
    rm -rf storage/
    tar -xzf "$backup_file"

    print_info "Starting service..."
    docker-compose up -d

    print_success "Restore completed successfully"
}

# Main command dispatcher
main() {
    check_docker

    case "${1:-}" in
        build)      cmd_build ;;
        start)      cmd_start ;;
        stop)       cmd_stop ;;
        restart)    cmd_restart ;;
        logs)       cmd_logs ;;
        status)     cmd_status ;;
        health)     cmd_health ;;
        shell)      cmd_shell ;;
        clean)      cmd_clean ;;
        rebuild)    cmd_rebuild ;;
        backup)     cmd_backup ;;
        restore)    cmd_restore "$2" ;;
        *)          show_usage ;;
    esac
}

main "$@"
