# Contributing to nano-banana-2-mcp

## Development Setup

```bash
git clone https://github.com/daveremy/nano-banana-2-mcp.git
cd nano-banana-2-mcp
npm install
```

## Scripts

| Command | Description |
|---------|-------------|
| `npm run build` | Compile TypeScript to `dist/` |
| `npm run dev` | Run from source with tsx (hot reload) |
| `npm test` | Run tests |
| `npm start` | Run compiled server |

## Running Locally

1. Build: `npm run build`
2. Set your API key: `export GEMINI_API_KEY=your-key`
3. Start: `npm start`

Or use `npm run dev` to run directly from source during development.

## Testing

```bash
npm test
```

Tests use Node's built-in test runner. Add new tests in `test/` with the `.test.ts` extension.

## Submitting Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Ensure `npm test` and `npm run build` pass
4. Submit a pull request

## Release Process

Releases are handled by maintainers using `npm run release`. See `scripts/release.sh` for details.
