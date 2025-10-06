#!/bin/bash
set -e

# MLX Distributed Inference Docker Build Script

# Configuration
REGISTRY=${REGISTRY:-"mlx-inference"}
VERSION=${VERSION:-"latest"}
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")}
GIT_BRANCH=${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")}

# Build arguments
BUILD_ARGS=(
    --build-arg BUILD_DATE="$BUILD_DATE"
    --build-arg GIT_COMMIT="$GIT_COMMIT"
    --build-arg GIT_BRANCH="$GIT_BRANCH"
    --build-arg VERSION="$VERSION"
)

# Images to build
IMAGES=(
    "api:docker/Dockerfile.api"
    "coordinator:docker/Dockerfile.coordinator"
    "monitoring:docker/Dockerfile.monitoring"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to build a single image
build_image() {
    local service=$1
    local dockerfile=$2
    local image_name="${REGISTRY}/${service}"
    local full_tag="${image_name}:${VERSION}"
    local latest_tag="${image_name}:latest"
    
    log "Building ${service} image..."
    log "Dockerfile: ${dockerfile}"
    log "Tag: ${full_tag}"
    
    # Build the image
    if docker build \
        "${BUILD_ARGS[@]}" \
        --tag "$full_tag" \
        --tag "$latest_tag" \
        --file "$dockerfile" \
        --label "org.opencontainers.image.created=$BUILD_DATE" \
        --label "org.opencontainers.image.revision=$GIT_COMMIT" \
        --label "org.opencontainers.image.version=$VERSION" \
        --label "org.opencontainers.image.source=https://github.com/mlx-distributed-inference" \
        --label "org.opencontainers.image.title=MLX Distributed Inference - $service" \
        --label "org.opencontainers.image.description=MLX Distributed Inference $service service" \
        .; then
        success "Successfully built ${service} image: ${full_tag}"
        return 0
    else
        error "Failed to build ${service} image"
        return 1
    fi
}

# Function to scan image for vulnerabilities (if trivy is available)
scan_image() {
    local image=$1
    
    if command -v trivy >/dev/null 2>&1; then
        log "Scanning ${image} for vulnerabilities..."
        if trivy image --exit-code 1 --severity HIGH,CRITICAL "$image"; then
            success "Security scan passed for ${image}"
        else
            warning "Security scan found issues in ${image}"
        fi
    else
        warning "Trivy not found, skipping security scan"
    fi
}

# Function to push image to registry
push_image() {
    local service=$1
    local image_name="${REGISTRY}/${service}"
    local full_tag="${image_name}:${VERSION}"
    local latest_tag="${image_name}:latest"
    
    if [ "$PUSH_IMAGES" = "true" ]; then
        log "Pushing ${service} image to registry..."
        
        if docker push "$full_tag" && docker push "$latest_tag"; then
            success "Successfully pushed ${service} image"
        else
            error "Failed to push ${service} image"
            return 1
        fi
    fi
}

# Main build function
main() {
    log "Starting MLX Distributed Inference Docker build"
    log "Registry: ${REGISTRY}"
    log "Version: ${VERSION}"
    log "Build Date: ${BUILD_DATE}"
    log "Git Commit: ${GIT_COMMIT}"
    log "Git Branch: ${GIT_BRANCH}"
    
    # Check if Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "docker-compose.yml" ]; then
        error "docker-compose.yml not found. Please run this script from the project root."
        exit 1
    fi
    
    # Build images
    local failed_builds=()
    
    for image_def in "${IMAGES[@]}"; do
        IFS=':' read -r service dockerfile <<< "$image_def"
        
        if build_image "$service" "$dockerfile"; then
            # Scan for vulnerabilities
            scan_image "${REGISTRY}/${service}:${VERSION}"
            
            # Push if requested
            push_image "$service"
        else
            failed_builds+=("$service")
        fi
    done
    
    # Report results
    if [ ${#failed_builds[@]} -eq 0 ]; then
        success "All images built successfully!"
        
        log "Built images:"
        for image_def in "${IMAGES[@]}"; do
            IFS=':' read -r service dockerfile <<< "$image_def"
            echo "  - ${REGISTRY}/${service}:${VERSION}"
        done
        
        log "To run the stack: docker-compose up -d"
        log "To run with production config: docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d"
        
    else
        error "Failed to build images: ${failed_builds[*]}"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --push)
            PUSH_IMAGES="true"
            shift
            ;;
        --no-cache)
            BUILD_ARGS+=(--no-cache)
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --registry REGISTRY    Docker registry prefix (default: mlx-inference)"
            echo "  --version VERSION      Image version tag (default: latest)"
            echo "  --push                 Push images to registry after building"
            echo "  --no-cache             Build without using cache"
            echo "  --help                 Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  REGISTRY              Docker registry prefix"
            echo "  VERSION               Image version tag"
            echo "  PUSH_IMAGES           Set to 'true' to push images"
            echo "  GIT_COMMIT            Git commit hash (auto-detected)"
            echo "  GIT_BRANCH            Git branch name (auto-detected)"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main