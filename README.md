# nano-banana-2-mcp

MCP server for Google Gemini image generation, upgraded for **Nano Banana 2** (`gemini-3.1-flash-image-preview`).

A fork/rewrite of [ConechoAI/Nano-Banana-MCP](https://github.com/ConechoAI/Nano-Banana-MCP) with:

- **Nano Banana 2 model** — better text rendering, thinking modes, resolution control
- **Resolution control** — 0.5K, 1K, 2K, 4K
- **Aspect ratio** — any ratio the API supports (1:1, 16:9, 9:16, 4:3, etc.)
- **Thinking modes** — minimal or high
- **Multiple images** — generate 1–4 variations per call
- **Output format** — PNG or JPEG
- **File-path-only mode** — no inline base64, fixes context window overflow in Claude Code
- **Security hardening** — path validation, file size caps, no plaintext API key storage

## Setup

### 1. Get a Gemini API key

Get one from [Google AI Studio](https://aistudio.google.com/apikey).

### 2. Add to Claude Code

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "nano-banana-2": {
      "command": "node",
      "args": ["/path/to/nano-banana-2-mcp/dist/index.js"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### 3. Build

```bash
cd nano-banana-2-mcp
npm install
npm run build
```

### 4. Restart Claude Code

The tools will be available after restart.

## Tools

### `generate_image`

Generate a new image from a text prompt.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | (required) | Text prompt for the image |
| `aspectRatio` | string | `"1:1"` | Aspect ratio (e.g. "16:9", "9:16") |
| `resolution` | string | `"1K"` | `0.5K`, `1K`, `2K`, `4K` |
| `thinking` | string | `"minimal"` | `minimal` or `high` |
| `numberOfImages` | number | `1` | 1–4 |
| `outputMimeType` | string | `"image/png"` | `image/png` or `image/jpeg` |
| `returnInlineImage` | boolean | `true` | If false, return only file path |

### `edit_image`

Edit an existing image file.

Same parameters as `generate_image`, plus:

| Parameter | Type | Description |
|-----------|------|-------------|
| `imagePath` | string | (required) Path to the image to edit |
| `referenceImages` | string[] | Optional additional reference images |

### `continue_editing`

Continue editing the last generated/edited image. Same parameters as `edit_image` minus `imagePath` (uses the last image automatically).

### `get_configuration_status`

Check API key, active model, and settings.

### `get_last_image_info`

Get path and size of the last generated image.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `NANO_BANANA_MODEL` | `gemini-3.1-flash-image-preview` | Model ID override |
| `NANO_BANANA_OUTPUT_DIR` | `./generated_imgs` | Image save directory |
| `NANO_BANANA_INLINE_IMAGE` | `true` | Default for `returnInlineImage` |

## Capability Gating

The server auto-detects model capabilities:

- **Image models** (`*-image`, `*-image-preview`): get `imageConfig` (resolution, aspect ratio, numberOfImages, outputMimeType)
- **Gemini 3.x image models**: also get `thinkingConfig`
- **Other models**: basic config only

This means you can use `NANO_BANANA_MODEL=gemini-2.5-flash-image` and it will send `imageConfig` but skip `thinkingConfig`.

## Claude Code Plugin

This repo includes a Claude Code plugin with a `generate-image` skill that provides best-practice prompting guidance. To use it, add the repo path to your Claude Code plugins config.

## Attribution

Based on [ConechoAI/Nano-Banana-MCP](https://github.com/ConechoAI/Nano-Banana-MCP) (MIT License).

## License

MIT
