#!/bin/bash
# SPDX-License-Identifier: Apache 2.0
# Copyright (c) 2025 IBM

# Maestro Knowledge MCP Server Stop Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use environment variables for runtime directories (OpenShift compatible)
# Falls back to SCRIPT_DIR for local development
RUNTIME_DIR="${RUNTIME_DIR:-$SCRIPT_DIR}"
LOG_DIR="${LOG_DIR:-$RUNTIME_DIR}"
PID_DIR="${PID_DIR:-$RUNTIME_DIR}"

PID_FILE="$PID_DIR/mcp_server.pid"
LOG_FILE="$LOG_DIR/mcp_server.log"
DEFAULT_PORT="8030"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${PURPLE}[INFO]${NC} $1"
}

# Function to check if a port is in use and kill the process
kill_port_process() {
    local port=$1
    local service_name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Killing $service_name process on port $port..."
        lsof -ti :$port | xargs kill -9 2>/dev/null
        sleep 1
    fi
}

# Function to identify service from PID
identify_service() {
    local pid=$1
    if ! kill -0 $pid 2>/dev/null; then
        echo "stale"
        return
    fi
    
    # Get the command line for the PID
    local cmd=$(ps -p $pid -o command= 2>/dev/null)
    if [[ $cmd == *"src.maestro_mcp.server"* ]]; then
        echo "MCP Server"
    elif [[ $cmd == *"python"* ]] && [[ $cmd == *"mcp"* ]]; then
        echo "MCP Server"
    else
        echo "Unknown"
    fi
}

# Check if server is running
check_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        # Check if the content is "ready" (stdio mode)
        if [ "$pid" = "ready" ]; then
            return 1  # Not a running HTTP server
        fi
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # Server is running
        else
            # PID file exists but process is dead
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1  # Server is not running
}

# Check if server is ready (for stdio mode)
check_ready() {
    if [ -f "$PID_FILE" ]; then
        local status=$(cat "$PID_FILE")
        if [ "$status" = "ready" ]; then
            return 0  # Server is ready
        else
            # Status file exists but with wrong status
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1  # Server is not ready
}

# Function to clean up stale PIDs from PID file
cleanup_stale_pids() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if [ "$pid" = "ready" ]; then
            # This is a status file, not a PID file
            return
        fi
        
        if ! kill -0 $pid 2>/dev/null; then
            echo "🧹 Cleaning up stale PID: $pid"
            rm -f "$PID_FILE"
            echo "✅ PID file cleaned up"
        fi
    fi
}

# Stop the MCP server
stop_server() {
    print_status "Stopping Maestro Knowledge MCP Server..."
    
    # Check if HTTP server is running
    if check_running; then
        local pid=$(cat "$PID_FILE")
        print_status "Found running HTTP server (PID: $pid)"
        
        # Attempt graceful shutdown
        print_status "Attempting graceful shutdown of HTTP server (PID: $pid)..."
        kill "$pid" 2>/dev/null # Send SIGTERM

        # Wait and check if process is still running
        local max_attempts=10
        local attempt=0
        while ps -p "$pid" > /dev/null 2>&1 && [ $attempt -lt $max_attempts ]; do
            sleep 1
            attempt=$((attempt+1))
        done

        if ps -p "$pid" > /dev/null 2>&1; then
            print_warning "HTTP server (PID: $pid) did not stop gracefully after $attempt seconds, force killing..."
            kill -9 "$pid"
        else
            print_success "HTTP server stopped successfully"
        fi
        
        # Remove the PID file
        rm -f "$PID_FILE"
        
        return 0
    fi
    
    # Check if stdio server is ready
    if check_ready; then
        local status=$(cat "$PID_FILE")
        print_status "Found ready stdio server (Status: $status)"
        
        # Remove the status file
        rm -f "$PID_FILE"
        print_success "MCP stdio server status cleared"
        return 0
    fi

    # Always kill any processes using default port
    kill_port_process $DEFAULT_PORT "MCP Server"
    
    print_warning "No MCP server was found running."
    return 0
}

# Show server status
show_status() {
    echo "=== Maestro Knowledge MCP Server Status ==="
    echo ""
    
    # Clean up stale PIDs first
    cleanup_stale_pids
    
    # Check PID file
    if [ -f "$PID_FILE" ]; then
        echo "📄 PID File Status:"
        local pid=$(cat "$PID_FILE")
        if [ "$pid" = "ready" ]; then
            echo "   ✅ Stdio server is ready (Status: $pid)"
        else
            local service=$(identify_service $pid)
            if [ "$service" = "stale" ]; then
                echo "   ❌ Process $pid is not running (stale PID)"
            else
                echo "   ✅ Process $pid is running ($service)"
            fi
        fi
    else
        echo "📄 PID File Status: No PID file found"
    fi
    
    echo ""
    echo "🌐 Port Status:"
    
    # Check default port
    if lsof -Pi :$DEFAULT_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        local pid=$(lsof -ti :$DEFAULT_PORT)
        local service=$(identify_service $pid)
        echo "   ✅ MCP Server (port $DEFAULT_PORT) - Running (PID: $pid)"
    else
        echo "   ❌ MCP Server (port $DEFAULT_PORT) - Not running"
    fi
    
    echo ""
    
    # Show detailed status based on what's running
    if check_running; then
        local pid=$(cat "$PID_FILE")
        echo "🎉 MCP HTTP server is running!"
        echo "   • Server URL: http://localhost:$DEFAULT_PORT"
        echo "   • OpenAPI docs: http://localhost:$DEFAULT_PORT/docs"
        echo "   • ReDoc docs: http://localhost:$DEFAULT_PORT/redoc"
        echo "   • MCP endpoint: http://localhost:$DEFAULT_PORT/mcp/"
        if [ -f "$LOG_FILE" ]; then
            echo "   • Log file: $LOG_FILE"
        fi
    elif check_ready; then
        echo "🎉 MCP stdio server is ready!"
        echo "   • To use with MCP clients, run: python -m src.maestro_mcp.server"
        echo "   • 💡 Tip: Use './start.sh --http' to start HTTP server for browser access"
        if [ -f "$LOG_FILE" ]; then
            echo "   • Log file: $LOG_FILE"
        fi
    else
        echo "⚠️  No MCP server is running."
        echo "   Use './start.sh' to start the server."
    fi
}

# Clean up stale files
cleanup() {
    print_status "Cleaning up stale files..."
    
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if [ "$pid" = "ready" ]; then
            print_status "Found ready status file, removing..."
            rm -f "$PID_FILE"
            print_success "Removed ready status file"
        elif ! ps -p "$pid" > /dev/null 2>&1; then
            rm -f "$PID_FILE"
            print_success "Removed stale PID file"
        else
            print_warning "PID file contains running process, not removing"
        fi
    fi
    
    # Also clean up any processes on our default port
    kill_port_process $DEFAULT_PORT "MCP Server"
}

# Function to restart all MCP processes
restart_processes() {
    echo "🔄 Restarting MCP server..."
    
    # Stop existing processes
    if stop_server; then
        echo "✅ All processes stopped successfully"
    else
        echo "⚠️  Some processes may still be running, continuing with restart..."
    fi
    
    # Wait a moment for cleanup
    sleep 2
    
    # Start processes using start.sh
    echo "🚀 Starting MCP server..."
    if [ -f start.sh ]; then
        ./start.sh --http
    else
        echo "❌ Error: start.sh not found!"
        exit 1
    fi
}

# Main execution
main() {
    print_status "Maestro Knowledge MCP Server Manager"
    print_status "====================================="
    
    case "${1:-stop}" in
        "stop")
            stop_server
            ;;
        "status")
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        "restart")
            restart_processes
            ;;
        "restart-http")
            print_status "Restarting MCP HTTP server..."
            stop_server
            sleep 2
            ./start.sh --http
            ;;
        *)
            print_error "Unknown command: $1"
            print_status "Usage: $0 {stop|status|cleanup|restart|restart-http}"
            print_status "  stop         - Stop the MCP server"
            print_status "  status       - Show server status"
            print_status "  cleanup      - Clean up stale files"
            print_status "  restart      - Restart the MCP server"
            print_status "  restart-http - Restart the MCP HTTP server"
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 
