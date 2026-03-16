import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { VERSION } from "../src/version.js";

describe("smoke", () => {
  it("VERSION is a semver string", () => {
    assert.match(VERSION, /^\d+\.\d+\.\d+$/);
  });
});
