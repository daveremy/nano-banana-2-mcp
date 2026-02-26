---
name: generate-image
description: Generate images using nano-banana-2 MCP tools with best-practice prompting. Use this skill when the user asks to create, generate, or edit images.
---

# Image Generation with Nano Banana 2

Use the `generate_image`, `edit_image`, and `continue_editing` MCP tools from the `nano-banana-2-mcp` server.

## Prompting Best Practices

**Write narrative paragraphs, NOT comma-separated keyword lists.** Narrative prompts achieve ~94% coherence vs ~61% for keyword lists.

1. **Start with image type**: "Create an educational diagram showing...", "Generate a photorealistic photograph of...", "Design a flat-style icon depicting..."
2. **Keep text in images short** (<25 characters). Quote text exactly and describe font style.
3. **Specify layout explicitly**: side-by-side, top-to-bottom steps, centered with border, etc.
4. **Skip quality boosters** like "4k masterpiece", "highly detailed", "award-winning" — they add noise, not quality.
5. **Be specific about what you want**, not what you don't want. Positive descriptions work better than negations.

## Defaults

- **resolution**: `"1K"` — good balance of quality and speed. Use `"2K"` or `"4K"` for print or detail-heavy images.
- **aspectRatio**: `"1:1"` — change to `"16:9"` for landscapes/banners, `"9:16"` for mobile/portrait.
- **thinking**: `"minimal"` — use `"high"` for complex scenes with spatial relationships or text rendering.
- **returnInlineImage**: Consider setting to `false` in Claude Code to avoid context window overflow. The image is still saved to disk and can be viewed by the user.
- **outputMimeType**: `"image/png"` for quality, `"image/jpeg"` for smaller files.
- **numberOfImages**: `1` — use 2-4 when exploring variations.

## Workflow

1. **Generate**: Use `generate_image` with a well-crafted narrative prompt.
2. **Review**: Check the saved file path in the response.
3. **Refine**: Use `continue_editing` to make adjustments. Be specific about what to change.
4. **Iterate**: Each `continue_editing` call builds on the previous result.

## Style Guidance

- **Diagrams**: Specify colors, label positions, arrow directions. Use `thinking: "high"` for complex layouts.
- **Illustrations**: Describe art style (flat, watercolor, line art), mood, and lighting.
- **Infographics**: Use numbered lists cautiously (>8 sequential items are unreliable). Numbers-only approach works best for long lists.
- **Photos**: Describe camera angle, lighting conditions, depth of field, and subject positioning.

## Context Window Management

When generating images in Claude Code, set `returnInlineImage: false` to prevent base64 image data from filling the context window. The image is still saved to disk — just tell the user where to find it.
