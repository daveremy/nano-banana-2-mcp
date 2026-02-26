#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
  CallToolRequest,
  CallToolResult,
  ErrorCode,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import { GoogleGenAI } from "@google/genai";
import fs from "fs/promises";
import path from "path";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_MODEL = "gemini-3.1-flash-image-preview";
const IMAGE_MODEL_PATTERN = /-image(-preview)?$/;
const GEMINI_3X_PATTERN = /gemini-3\./;

const VALID_RESOLUTIONS = new Set(["0.5K", "1K", "2K", "4K"]);
const VALID_THINKING = new Set(["minimal", "high"]);
const VALID_MIME_TYPES = new Set(["image/png", "image/jpeg"]);
const ALLOWED_IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp"]);
const MAX_IMAGE_FILE_SIZE = 50 * 1024 * 1024; // 50 MB
const MAX_NUMBER_OF_IMAGES = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getModelId(): string {
  return process.env.NANO_BANANA_MODEL || DEFAULT_MODEL;
}

function isImageModel(modelId: string): boolean {
  return IMAGE_MODEL_PATTERN.test(modelId);
}

function supportsThinking(modelId: string): boolean {
  return isImageModel(modelId) && GEMINI_3X_PATTERN.test(modelId);
}

function getOutputDir(): string {
  return process.env.NANO_BANANA_OUTPUT_DIR || path.join(process.cwd(), "generated_imgs");
}

function resolveInlineImage(perCall: boolean | undefined): boolean {
  if (perCall !== undefined) return perCall;
  const env = process.env.NANO_BANANA_INLINE_IMAGE;
  if (env !== undefined) return env === "true";
  return true; // default
}

function mimeToExtension(mime: string): string {
  return mime === "image/jpeg" ? ".jpg" : ".png";
}

function randomId(): string {
  return Math.random().toString(36).slice(2, 8);
}

function timestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function validateImagePath(filePath: string): Promise<void> {
  const ext = path.extname(filePath).toLowerCase();
  if (!ALLOWED_IMAGE_EXTENSIONS.has(ext)) {
    throw new McpError(
      ErrorCode.InvalidParams,
      `Invalid image extension "${ext}". Allowed: ${[...ALLOWED_IMAGE_EXTENSIONS].join(", ")}`,
    );
  }

  let stat;
  try {
    stat = await fs.stat(filePath);
  } catch {
    throw new McpError(ErrorCode.InvalidParams, `File not found: ${filePath}`);
  }

  if (!stat.isFile()) {
    throw new McpError(ErrorCode.InvalidParams, `Not a file: ${filePath}`);
  }

  if (stat.size > MAX_IMAGE_FILE_SIZE) {
    throw new McpError(
      ErrorCode.InvalidParams,
      `File too large (${formatBytes(stat.size)}). Max: ${formatBytes(MAX_IMAGE_FILE_SIZE)}`,
    );
  }
}

// ---------------------------------------------------------------------------
// Common image generation parameters (shared schema)
// ---------------------------------------------------------------------------

const imageParamProperties = {
  aspectRatio: {
    type: "string" as const,
    description: "Aspect ratio (e.g. \"1:1\", \"16:9\", \"9:16\", \"4:3\", \"3:4\"). Passed through to API.",
    default: "1:1",
  },
  resolution: {
    type: "string" as const,
    description: "Image resolution: \"0.5K\", \"1K\", \"2K\", or \"4K\".",
    default: "1K",
  },
  thinking: {
    type: "string" as const,
    description: "Thinking level: \"minimal\" or \"high\". Higher thinking improves complex prompts.",
    default: "minimal",
  },
  numberOfImages: {
    type: "number" as const,
    description: "Number of images to generate (1–4). Multiple images saved with sequential suffixes.",
    default: 1,
  },
  outputMimeType: {
    type: "string" as const,
    description: "Output format: \"image/png\" or \"image/jpeg\".",
    default: "image/png",
  },
  returnInlineImage: {
    type: "boolean" as const,
    description: "If false, return only file path (no base64). Saves context window space.",
  },
};

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

class NanoBanana2MCP {
  private server: Server;
  private genAI: GoogleGenAI | null = null;
  private lastImagePath: string | null = null;

  constructor() {
    this.server = new Server(
      { name: "nano-banana-2-mcp", version: "1.0.0" },
      { capabilities: { tools: {} } },
    );
    this.setupHandlers();
  }

  // -------------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------------

  private initGenAI(): GoogleGenAI {
    if (this.genAI) return this.genAI;
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new McpError(
        ErrorCode.InvalidRequest,
        "GEMINI_API_KEY environment variable is required. Set it in your MCP server config.",
      );
    }
    this.genAI = new GoogleGenAI({ apiKey });
    return this.genAI;
  }

  // -------------------------------------------------------------------------
  // Tool definitions
  // -------------------------------------------------------------------------

  private setupHandlers(): void {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: "generate_image",
          description:
            "Generate a NEW image from text prompt. Use this ONLY when creating a completely new image, not when modifying an existing one.",
          inputSchema: {
            type: "object",
            properties: {
              prompt: {
                type: "string",
                description: "Text prompt describing the NEW image to create from scratch",
              },
              ...imageParamProperties,
            },
            required: ["prompt"],
          },
        },
        {
          name: "edit_image",
          description:
            "Edit a SPECIFIC existing image file, optionally using additional reference images. Use this when you have the exact file path of an image to modify.",
          inputSchema: {
            type: "object",
            properties: {
              imagePath: {
                type: "string",
                description: "Full file path to the main image file to edit",
              },
              prompt: {
                type: "string",
                description: "Text describing the modifications to make to the existing image",
              },
              referenceImages: {
                type: "array",
                items: { type: "string" },
                description: "Optional array of file paths to additional reference images",
              },
              ...imageParamProperties,
            },
            required: ["imagePath", "prompt"],
          },
        },
        {
          name: "continue_editing",
          description:
            "Continue editing the LAST image that was generated or edited in this session, optionally using additional reference images.",
          inputSchema: {
            type: "object",
            properties: {
              prompt: {
                type: "string",
                description: "Text describing the modifications to the last image",
              },
              referenceImages: {
                type: "array",
                items: { type: "string" },
                description: "Optional array of file paths to additional reference images",
              },
              ...imageParamProperties,
            },
            required: ["prompt"],
          },
        },
        {
          name: "get_configuration_status",
          description: "Check if Gemini API key is configured and which model is active",
          inputSchema: { type: "object", properties: {}, additionalProperties: false },
        },
        {
          name: "get_last_image_info",
          description: "Get information about the last generated/edited image in this session",
          inputSchema: { type: "object", properties: {}, additionalProperties: false },
        },
      ] as Tool[],
    }));

    this.server.setRequestHandler(
      CallToolRequestSchema,
      async (request: CallToolRequest): Promise<CallToolResult> => {
        try {
          switch (request.params.name) {
            case "generate_image":
              return await this.generateImage(request);
            case "edit_image":
              return await this.editImage(request);
            case "continue_editing":
              return await this.continueEditing(request);
            case "get_configuration_status":
              return this.getConfigurationStatus();
            case "get_last_image_info":
              return await this.getLastImageInfo();
            default:
              throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${request.params.name}`);
          }
        } catch (error) {
          if (error instanceof McpError) throw error;
          throw new McpError(
            ErrorCode.InternalError,
            `Tool execution failed: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      },
    );
  }

  // -------------------------------------------------------------------------
  // Parameter extraction
  // -------------------------------------------------------------------------

  private extractImageParams(args: Record<string, unknown>) {
    const aspectRatio = (args.aspectRatio as string) || "1:1";
    const resolution = (args.resolution as string) || "1K";
    const thinking = (args.thinking as string) || "minimal";
    const numberOfImages = Math.min(Math.max(Math.round(Number(args.numberOfImages) || 1), 1), MAX_NUMBER_OF_IMAGES);
    const outputMimeType = (args.outputMimeType as string) || "image/png";
    const returnInlineImage = resolveInlineImage(
      args.returnInlineImage === undefined ? undefined : Boolean(args.returnInlineImage),
    );

    // Validate
    if (!VALID_RESOLUTIONS.has(resolution)) {
      throw new McpError(ErrorCode.InvalidParams, `Invalid resolution "${resolution}". Use: ${[...VALID_RESOLUTIONS].join(", ")}`);
    }
    if (!VALID_THINKING.has(thinking)) {
      throw new McpError(ErrorCode.InvalidParams, `Invalid thinking "${thinking}". Use: ${[...VALID_THINKING].join(", ")}`);
    }
    if (!VALID_MIME_TYPES.has(outputMimeType)) {
      throw new McpError(ErrorCode.InvalidParams, `Invalid outputMimeType "${outputMimeType}". Use: ${[...VALID_MIME_TYPES].join(", ")}`);
    }

    return { aspectRatio, resolution, thinking, numberOfImages, outputMimeType, returnInlineImage };
  }

  // -------------------------------------------------------------------------
  // Gemini API call
  // -------------------------------------------------------------------------

  private async callGemini(
    contents: string | Array<{ parts: Array<Record<string, unknown>> }>,
    params: ReturnType<typeof this.extractImageParams>,
  ) {
    const genAI = this.initGenAI();
    const modelId = getModelId();
    const imageModel = isImageModel(modelId);
    const thinkingSupported = supportsThinking(modelId);

    const config: Record<string, unknown> = {
      responseModalities: ["IMAGE"],
      ...(imageModel && {
        imageConfig: {
          aspectRatio: params.aspectRatio,
          imageSize: params.resolution,
          numberOfImages: params.numberOfImages,
          outputMimeType: params.outputMimeType,
        },
      }),
      ...(thinkingSupported && {
        thinkingConfig: {
          thinkingLevel: params.thinking,
        },
      }),
    };

    const response = await genAI.models.generateContent({
      model: modelId,
      contents,
      config,
    });

    return response;
  }

  // -------------------------------------------------------------------------
  // Image saving
  // -------------------------------------------------------------------------

  private async ensureOutputDir(): Promise<string> {
    const dir = getOutputDir();
    await fs.mkdir(dir, { recursive: true });
    return dir;
  }

  private async saveImage(
    base64Data: string,
    mimeType: string,
    prefix: string,
    suffix?: string,
  ): Promise<{ filePath: string; fileSize: number }> {
    const dir = await this.ensureOutputDir();
    const ext = mimeToExtension(mimeType);
    const name = `${prefix}-${timestamp()}-${randomId()}${suffix || ""}${ext}`;
    const filePath = path.join(dir, name);
    const buffer = Buffer.from(base64Data, "base64");
    await fs.writeFile(filePath, buffer);
    return { filePath, fileSize: buffer.length };
  }

  // -------------------------------------------------------------------------
  // Response building
  // -------------------------------------------------------------------------

  private buildResponse(
    savedImages: Array<{ filePath: string; fileSize: number; base64: string; mimeType: string }>,
    returnInlineImage: boolean,
  ): CallToolResult {
    const content: Array<{ type: string; text?: string; data?: string; mimeType?: string }> = [];

    // Text summary
    const lines: string[] = [];
    for (const img of savedImages) {
      lines.push(`${img.filePath} (${formatBytes(img.fileSize)})`);
    }
    if (savedImages.length === 1) {
      lines.push("Use continue_editing to refine this image.");
    } else {
      lines.push("Use continue_editing to refine the first image.");
    }
    content.push({ type: "text", text: lines.join("\n") });

    // Inline images (if enabled)
    if (returnInlineImage) {
      for (const img of savedImages) {
        content.push({
          type: "image",
          data: img.base64,
          mimeType: img.mimeType,
        });
      }
    }

    return { content } as CallToolResult;
  }

  // -------------------------------------------------------------------------
  // Tools: generate_image
  // -------------------------------------------------------------------------

  private async generateImage(request: CallToolRequest): Promise<CallToolResult> {
    const args = request.params.arguments as Record<string, unknown>;
    const prompt = args.prompt as string;
    if (!prompt) throw new McpError(ErrorCode.InvalidParams, "prompt is required");

    const params = this.extractImageParams(args);
    const response = await this.callGemini(prompt, params);

    const savedImages = await this.processResponse(response, "generated", params);
    if (savedImages.length === 0) {
      return { content: [{ type: "text", text: "No image was generated. Try rephrasing your prompt." }] };
    }

    this.lastImagePath = savedImages[0].filePath;
    return this.buildResponse(savedImages, params.returnInlineImage);
  }

  // -------------------------------------------------------------------------
  // Tools: edit_image
  // -------------------------------------------------------------------------

  private async editImage(request: CallToolRequest): Promise<CallToolResult> {
    const args = request.params.arguments as Record<string, unknown>;
    const imagePath = args.imagePath as string;
    const prompt = args.prompt as string;
    const referenceImages = (args.referenceImages as string[]) || [];

    if (!imagePath) throw new McpError(ErrorCode.InvalidParams, "imagePath is required");
    if (!prompt) throw new McpError(ErrorCode.InvalidParams, "prompt is required");

    // Validate all image paths
    const allPaths = [imagePath, ...referenceImages];
    for (const p of allPaths) {
      await validateImagePath(p);
    }

    // Build parts: images first, then text prompt
    const parts: Array<Record<string, unknown>> = [];
    for (const p of allPaths) {
      const data = await fs.readFile(p);
      const ext = path.extname(p).toLowerCase();
      const mime =
        ext === ".png" ? "image/png" :
        ext === ".webp" ? "image/webp" :
        "image/jpeg";
      parts.push({
        inlineData: {
          data: data.toString("base64"),
          mimeType: mime,
        },
      });
    }
    parts.push({ text: prompt });

    const params = this.extractImageParams(args);
    const response = await this.callGemini([{ parts }], params);

    const savedImages = await this.processResponse(response, "edited", params);
    if (savedImages.length === 0) {
      return { content: [{ type: "text", text: "No edited image was produced. Try a different prompt." }] };
    }

    this.lastImagePath = savedImages[0].filePath;
    return this.buildResponse(savedImages, params.returnInlineImage);
  }

  // -------------------------------------------------------------------------
  // Tools: continue_editing
  // -------------------------------------------------------------------------

  private async continueEditing(request: CallToolRequest): Promise<CallToolResult> {
    if (!this.lastImagePath) {
      throw new McpError(
        ErrorCode.InvalidRequest,
        "No previous image in this session. Use generate_image or edit_image first.",
      );
    }

    const args = request.params.arguments as Record<string, unknown>;
    // Delegate to editImage with lastImagePath
    const editArgs = { ...args, imagePath: this.lastImagePath };
    const editRequest = {
      ...request,
      params: { ...request.params, arguments: editArgs },
    } as CallToolRequest;

    return this.editImage(editRequest);
  }

  // -------------------------------------------------------------------------
  // Tools: get_configuration_status
  // -------------------------------------------------------------------------

  private getConfigurationStatus(): CallToolResult {
    const hasKey = !!process.env.GEMINI_API_KEY;
    const modelId = getModelId();
    const lines = [
      `API key: ${hasKey ? "configured" : "NOT configured — set GEMINI_API_KEY in MCP server env"}`,
      `Model: ${modelId}`,
      `Image model: ${isImageModel(modelId)}`,
      `Thinking support: ${supportsThinking(modelId)}`,
      `Output dir: ${getOutputDir()}`,
      `Inline images: ${resolveInlineImage(undefined)}`,
    ];
    return { content: [{ type: "text", text: lines.join("\n") }] };
  }

  // -------------------------------------------------------------------------
  // Tools: get_last_image_info
  // -------------------------------------------------------------------------

  private async getLastImageInfo(): Promise<CallToolResult> {
    if (!this.lastImagePath) {
      return { content: [{ type: "text", text: "No image generated in this session yet." }] };
    }

    try {
      const stat = await fs.stat(this.lastImagePath);
      return {
        content: [
          {
            type: "text",
            text: `Last image: ${this.lastImagePath}\nSize: ${formatBytes(stat.size)}`,
          },
        ],
      };
    } catch {
      return {
        content: [
          {
            type: "text",
            text: `Last image path recorded: ${this.lastImagePath}\n(File may have been moved or deleted)`,
          },
        ],
      };
    }
  }

  // -------------------------------------------------------------------------
  // Process Gemini response → saved images
  // -------------------------------------------------------------------------

  private async processResponse(
    response: Awaited<ReturnType<GoogleGenAI["models"]["generateContent"]>>,
    prefix: string,
    params: ReturnType<typeof this.extractImageParams>,
  ): Promise<Array<{ filePath: string; fileSize: number; base64: string; mimeType: string }>> {
    const candidates = response.candidates || [];
    const saved: Array<{ filePath: string; fileSize: number; base64: string; mimeType: string }> = [];
    let imageIndex = 0;

    for (const candidate of candidates) {
      const parts = candidate.content?.parts || [];
      for (const part of parts) {
        if (part.inlineData?.data) {
          imageIndex++;
          const suffix = params.numberOfImages > 1 ? `-${imageIndex}` : "";
          const mimeType = part.inlineData.mimeType || params.outputMimeType;
          const { filePath, fileSize } = await this.saveImage(
            part.inlineData.data,
            mimeType,
            prefix,
            suffix,
          );
          saved.push({
            filePath,
            fileSize,
            base64: part.inlineData.data,
            mimeType,
          });
        }
      }
    }

    return saved;
  }

  // -------------------------------------------------------------------------
  // Run
  // -------------------------------------------------------------------------

  public async run(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
  }
}

const server = new NanoBanana2MCP();
server.run().catch(console.error);
