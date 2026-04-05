# Multi-stage build for security and size optimization
FROM node:20-alpine AS deps
WORKDIR /app

# Install dependencies only (cached layer)
COPY package.json package-lock.json* ./
RUN npm ci --legacy-peer-deps --only=production && \
    npm cache clean --force

# Build stage
FROM node:20-alpine AS builder
WORKDIR /app

# Install ALL dependencies including devDependencies for build
COPY package.json package-lock.json* ./
RUN npm install --legacy-peer-deps

# Copy source code
COPY . .

# Build environment variables
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production

# Build application
RUN npm run build

# Production stage
FROM node:20-alpine AS runner
WORKDIR /app

# Security: Run as non-root user
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Create system user and group
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Security: Set proper file permissions
RUN mkdir -p /app/.next && \
    chown -R nextjs:nodejs /app

# Copy only necessary files
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Security: Remove unnecessary files
RUN rm -rf /app/.env* /app/README.md /app/CLAUDE.md 2>/dev/null || true

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Runtime environment
ENV HOSTNAME="0.0.0.0"
ENV PORT=3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/api/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})" || exit 1

# Start application
CMD ["node", "server.js"]