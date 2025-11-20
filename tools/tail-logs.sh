#!/bin/bash
# SPDX-License-Identifier: Apache 2.0
# Copyright (c) 2025 IBM

# Maestro Knowledge Log Tailing Script
# Usage: ./tools/tail-logs.sh [all|mcp|cli|status|recent]

# Colors for different services
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use environment variables for runtime directories (OpenShift compatible)
# Falls back to PROJECT_ROOT for local development
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT}"
LOG_DIR="${LOG_DIR:-$RUNTIME_DIR}"
PID_DIR="${PID_DIR:-$RUNTIME_DIR}"

# Check multiple locations for log files (OpenShift /tmp or local PROJECT_ROOT)
if [ -f "$LOG_DIR/mcp_server.log" ]; then
    MCP_LOG_FILE="$LOG_DIR/mcp_server.log"
elif [ -f "$PROJECT_ROOT/mcp_server.log" ]; then
    MCP_LOG_FILE="$PROJECT_ROOT/mcp_server.log"
else
    MCP_LOG_FILE="$LOG_DIR/mcp_server.log"  # Default to LOG_DIR
fi

if [ -f "$PID_DIR/mcp_server.pid" ]; then
    MCP_PID_FILE="$PID_DIR/mcp_server.pid"
elif [ -f "$PROJECT_ROOT/mcp_server.pid" ]; then
    MCP_PID_FILE="$PROJECT_ROOT/mcp_server.pid"
else
    MCP_PID_FILE="$PID_DIR/mcp_server.pid"  # Default to PID_DIR
fi

DEFAULT_PORT="8030"

# Function to check if a service is running
check_service() {
    local port=$1
    local service_name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${GREEN}✅ $service_name is running on port $port${NC}"
        return 0
    else
        echo -e "${RED}❌ $service_name is not running on port $port${NC}"
        return 1
    fi
}

# Function to get the MCP server PID
get_mcp_pid() {
    if [ -f "$MCP_PID_FILE" ]; then
        local pid=$(cat "$MCP_PID_FILE")
        if [ "$pid" != "ready" ] && ps -p "$pid" > /dev/null 2>&1; then
            echo "$pid"
        fi
    fi
}

# Function to show all service status
show_status() {
    echo -e "${BLUE}=== Maestro Knowledge Service Status ===${NC}"
    echo ""
    
    # Check MCP server
    check_service $DEFAULT_PORT "MCP Server"
    
    # Check MCP process
    local mcp_pid=$(get_mcp_pid)
    if [ ! -z "$mcp_pid" ]; then
        echo -e "${GREEN}✅ MCP Server process is running (PID: $mcp_pid)${NC}"
    else
        echo -e "${RED}❌ MCP Server process is not running${NC}"
    fi
    
    # Check if stdio server is ready
    if [ -f "$MCP_PID_FILE" ]; then
        local status=$(cat "$MCP_PID_FILE")
        if [ "$status" = "ready" ]; then
            echo -e "${GREEN}✅ MCP Stdio server is ready${NC}"
        fi
    fi
    
    echo ""
    echo -e "${BLUE}Log Files:${NC}"
    if [ -f "$MCP_LOG_FILE" ]; then
        local log_size=$(du -h "$MCP_LOG_FILE" | cut -f1)
        echo -e "${GREEN}✅ MCP Server log: $MCP_LOG_FILE (${log_size})${NC}"
    else
        echo -e "${RED}❌ MCP Server log not found${NC}"
    fi
    
    # CLI logs are now in the separate repository: AI4quantum/maestro-cli
    echo -e "${BLUE}ℹ️  CLI logs are now in the separate repository: AI4quantum/maestro-cli${NC}"
}

# Function to show recent logs
show_recent_logs() {
    echo -e "${BLUE}=== Recent Maestro Knowledge Logs (Last 50 lines) ===${NC}"
    echo ""
    
    # Show recent MCP server logs
    if [ -f "$MCP_LOG_FILE" ]; then
        echo -e "${YELLOW}Recent MCP Server logs:${NC}"
        tail -50 "$MCP_LOG_FILE" | while read line; do
            echo "[MCP] $line"
        done
        echo ""
    fi
    
    # CLI logs are now in the separate repository: AI4quantum/maestro-cli
    echo -e "${BLUE}ℹ️  CLI logs are now in the separate repository: AI4quantum/maestro-cli${NC}"
        echo ""
    fi
    
    # Show recent system logs that might be related to Maestro Knowledge
    echo -e "${YELLOW}Recent system logs containing 'maestro', 'mcp', or 'python':${NC}"
    log show --predicate 'eventMessage CONTAINS "maestro" OR eventMessage CONTAINS "mcp" OR eventMessage CONTAINS "python"' --last 5m 2>/dev/null | tail -20 | while read line; do
        echo "[SYS] $line"
    done
}

# Function to tail MCP server logs
tail_mcp_logs() {
    echo -e "${BLUE}📡 Tailing MCP Server logs (port $DEFAULT_PORT)...${NC}"
    if check_service $DEFAULT_PORT "MCP Server"; then
        if [ -f "$MCP_LOG_FILE" ]; then
            echo -e "${BLUE}Following MCP Server log file...${NC}"
            echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
            echo ""
            tail -f "$MCP_LOG_FILE"
        else
            echo -e "${RED}❌ MCP Server log file not found ($MCP_LOG_FILE)${NC}"
            echo -e "${YELLOW}MCP Server may not be running or log file not created${NC}"
        fi
    fi
}

# Function to tail CLI logs
tail_cli_logs() {
    echo -e "${BLUE}📡 CLI logs are now in the separate repository: AI4quantum/maestro-cli${NC}"
    echo -e "${YELLOW}Please check the maestro-cli repository for CLI logs${NC}"
}

# Function to tail all logs
tail_all_logs() {
    echo -e "${BLUE}📡 Tailing all Maestro Knowledge logs...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
    echo ""
    
    # Check which log files exist
    local log_files=()
    
    if [ -f "$MCP_LOG_FILE" ]; then
        log_files+=("$MCP_LOG_FILE")
    fi
    
    # CLI logs are now in the separate repository: AI4quantum/maestro-cli
    
    if [ ${#log_files[@]} -gt 0 ]; then
        # Use tail -f with multiple log files
        tail -f "${log_files[@]}"
    else
        echo -e "${YELLOW}⚠️  No log files found. Services may not have created logs yet.${NC}"
        echo -e "${YELLOW}Available log files:${NC}"
        ls -la "$PROJECT_ROOT"/*.log 2>/dev/null || echo "No log files found in project root"
        ls -la "$PROJECT_ROOT/cli"/*.log 2>/dev/null || echo "No log files found in cli directory"
    fi
}

# Function to monitor system logs for Maestro Knowledge processes
monitor_system_logs() {
    local service_name=$1
    local service_tag=$2
    local pid=$3
    
    echo -e "${YELLOW}Monitoring system logs for $service_name (PID: $pid)...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
    echo ""
    
    # Monitor system logs for the specific process
    log stream --predicate "process == '$pid'" 2>/dev/null | while read line; do
        echo "[$service_tag] $line"
    done
}

# Function to show help
show_help() {
    echo -e "${BLUE}Maestro Knowledge Log Tailing Script${NC}"
    echo ""
    echo "Usage: $0 [all|mcp|cli|status|recent]"
    echo ""
    echo "Commands:"
    echo -e "  ${GREEN}all${NC}        - Tail logs from all running services"
    echo -e "  ${GREEN}mcp${NC}        - Tail MCP Server logs (port $DEFAULT_PORT)"
    echo -e "  ${GREEN}cli${NC}        - Tail CLI logs"
    echo -e "  ${GREEN}status${NC}     - Show status of all services"
    echo -e "  ${GREEN}recent${NC}     - Show recent logs"
    echo ""
    echo "Examples:"
    echo "  $0 all           # Tail all service logs"
    echo "  $0 mcp           # Tail only MCP Server logs"
    echo "  $0 cli           # Tail only CLI logs"
    echo "  $0 status        # Show service status"
    echo "  $0 recent        # Show recent logs"
    echo ""
    echo "Note: This script tails real-time logs from Maestro Knowledge services."
    echo "Make sure services are running with './start.sh' before tailing logs."
}

# Main script logic
case "${1:-all}" in
    "all")
        tail_all_logs
        ;;
    "mcp")
        tail_mcp_logs
        ;;
    "cli")
        tail_cli_logs
        ;;
    "status")
        show_status
        ;;
    "recent")
        show_recent_logs
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac 