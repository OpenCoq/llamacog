# RR (Relevance Realization) Reasoning Format

This document describes the newly implemented RR (Relevance Realization) reasoning format in llamacog, based on the triadic architecture of relevance realization.

## Overview

The RR reasoning format extends the existing reasoning capabilities of llama.cpp to support triadic reasoning architectures as described in cognitive science research on relevance realization. This format uses `<rr></rr>` tags to delimit reasoning content.

## Usage

The RR reasoning format can be used in the same way as other reasoning formats:

### Basic Usage
- **Tag format**: `<rr>reasoning content</rr>`
- **Enum value**: `COMMON_REASONING_FORMAT_RR`
- **Format name**: `"rr"`

### Configuration Options

The RR format supports the same configuration options as other reasoning formats:

1. **reasoning_in_content = false** (default): Reasoning content is extracted to `message.reasoning_content`
2. **reasoning_in_content = true**: Reasoning content remains inline using `<rr></rr>` tags
3. **thinking_forced_open**: Allows parsing of unclosed reasoning tags

## Examples

### Example 1: Extract reasoning content
```cpp
common_chat_msg_parser builder("<rr>Triadic relevance analysis</rr>Response", false, {
    .format = COMMON_CHAT_FORMAT_CONTENT_ONLY,
    .reasoning_format = COMMON_REASONING_FORMAT_RR,
    .reasoning_in_content = false,
    .thinking_forced_open = false,
});

builder.try_parse_reasoning("<rr>", "</rr>");
// Result: 
// - builder.result().reasoning_content = "Triadic relevance analysis"
// - builder.consume_rest() = "Response"
```

### Example 2: Inline reasoning content
```cpp
common_chat_msg_parser builder("Triadic relevance analysis</rr>Response", false, {
    .format = COMMON_CHAT_FORMAT_CONTENT_ONLY,
    .reasoning_format = COMMON_REASONING_FORMAT_RR,
    .reasoning_in_content = true,
    .thinking_forced_open = true,
});

builder.try_parse_reasoning("<rr>", "</rr>");
// Result:
// - builder.result().content = "<rr>Triadic relevance analysis</rr>"
// - builder.consume_rest() = "Response"
```

## Implementation Details

The RR format is implemented with minimal changes to the existing reasoning framework:

1. **Enum addition**: Added `COMMON_REASONING_FORMAT_RR` to `common_reasoning_format`
2. **Name mapping**: Added "rr" case to `common_reasoning_format_name()`
3. **Parsing logic**: Enhanced `try_parse_reasoning()` to handle `<rr></rr>` tags
4. **Test coverage**: Added comprehensive test cases for all RR scenarios

## Triadic Architecture Support

The RR format is designed to support the triadic architecture of relevance realization, which involves:

- **Autopoiesis**: Self-creating dynamics
- **Anticipation**: Projective modeling
- **Adaptation**: Agent-arena dynamics

The `<rr></rr>` tags can contain reasoning content that encompasses these triadic relationships, enabling more sophisticated cognitive modeling in language models.

## Compatibility

The RR format is fully compatible with the existing reasoning framework and does not break any existing functionality. All previous reasoning formats (NONE, DEEPSEEK, DEEPSEEK_LEGACY) continue to work unchanged.