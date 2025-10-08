# OpenRouter Provider Selection Feature

## Overview
This document describes the new feature that allows users to specify their preferred upstream provider when using OpenRouter.

## GitHub Issue Reference
- **Issue #36**: "Allow choosing a specific provider on openrouter"
- **Requested by**: oogaboogaw

## Problem Statement
The user reported that Glossarion was using the `openrouter/deepseek/deepseek-chat-v3.1:free` model, which has two providers available through OpenRouter:
1. **Open Inference**
2. **DeepInfra**

Previously, Glossarion would send requests to OpenRouter, but OpenRouter would automatically choose which provider to route the request to. The user experienced an issue where:
- Glossarion always sent requests to **DeepInfra** until September 28th
- Then it suddenly switched to **Open Inference**
- Unfortunately, **Open Inference is heavily censored**, making it difficult to translate novels

The user asked for a way to specify which provider to use, or if there was already a setting for this.

## Solution Implemented

### 1. Updated API Safety Filter Label
**File**: `src/other_settings.py`

Changed the label from:
- **Old**: "Disable Gemini API Safety Filters"
- **New**: "Disable API Safety Filters (Gemini, Groq, Fireworks, etc.)"

This clarifies that the toggle affects multiple providers, not just Gemini:
- **Gemini**: Sets all harm categories to BLOCK_NONE
- **Groq/Fireworks**: Disables moderation parameter
- **Does NOT affect**: ElectronHub Gemini models (eh/gemini-*) or Together AI

### 2. Added OpenRouter Provider Preference UI
**File**: `src/other_settings.py`

Added a new dropdown menu below the "Use HTTP-only for OpenRouter" toggle with the following options:
- **Auto** (default - lets OpenRouter choose)
- **DeepInfra**
- **OpenInference**
- **Together**
- **Fireworks**
- **Lepton**
- **Mancer**

The UI includes a helpful description:
> "Specify which upstream provider OpenRouter should prefer for your requests. 'Auto' lets OpenRouter choose. Specific providers may have different availability."

### 3. Environment Variable Support
**File**: `src/translator_gui.py`

#### setup_environment() method
Added the `OPENROUTER_PREFERRED_PROVIDER` environment variable to the configuration:
```python
'OPENROUTER_PREFERRED_PROVIDER': self.config.get('openrouter_preferred_provider', 'Auto')
```

#### save_config() method
Added persistence for the provider preference:
```python
if hasattr(self, 'openrouter_preferred_provider_var'):
    self.config['openrouter_preferred_provider'] = self.openrouter_preferred_provider_var.get()
    os.environ['OPENROUTER_PREFERRED_PROVIDER'] = self.openrouter_preferred_provider_var.get()
```

### 4. OpenRouter API Integration
**File**: `src/unified_api_client.py`

Modified both the **SDK path** (using OpenAI SDK) and **HTTP path** (direct HTTP requests) to include the provider preference in the request body.

#### SDK Path (lines ~8606-8615)
```python
# Add provider preference if specified
try:
    preferred_provider = os.getenv('OPENROUTER_PREFERRED_PROVIDER', 'Auto').strip()
    if preferred_provider and preferred_provider != 'Auto':
        extra_body["provider"] = {
            "order": [preferred_provider]
        }
        print(f"🔀 OpenRouter: Requesting {preferred_provider} provider")
except Exception:
    pass
```

#### HTTP Path (lines ~8866-8875)
```python
# Add provider preference if specified
try:
    preferred_provider = os.getenv('OPENROUTER_PREFERRED_PROVIDER', 'Auto').strip()
    if preferred_provider and preferred_provider != 'Auto':
        data["provider"] = {
            "order": [preferred_provider]
        }
        print(f"🔀 OpenRouter: Requesting {preferred_provider} provider")
except Exception:
    pass
```

#### Debug Logging
The provider preference is also saved in the OpenRouter debug config file:
```python
# Persist provider preference
try:
    preferred_provider = os.getenv('OPENROUTER_PREFERRED_PROVIDER', 'Auto').strip()
    if preferred_provider and preferred_provider != 'Auto':
        cfg["preferred_provider"] = preferred_provider
except Exception:
    pass
```

## How to Use

1. **Open Glossarion**
2. **Navigate to Settings** → **Other Settings**
3. **Scroll to "API Safety Settings" section**
4. **Find "Preferred OpenRouter Provider" dropdown**
5. **Select your preferred provider** (e.g., "DeepInfra" to avoid Open Inference censorship)
6. **Save settings**
7. **Start translation** - Glossarion will now request your preferred provider from OpenRouter

## How to Verify the Provider

Glossarion now includes comprehensive logging to verify which provider OpenRouter actually used:

### Console Output
When a translation request is made with a preferred provider, you'll see:

1. **Request logging**:
   ```
   🔀 OpenRouter: Requesting DeepInfra provider
   ```

2. **Response logging** (when using HTTP-only mode):
   ```
   ✅ OpenRouter Provider (from header): DeepInfra
   📋 OpenRouter Response Model: deepseek/deepseek-chat-v3.1:free
   🆔 OpenRouter Generation ID: gen-abc123xyz
   📊 OpenRouter Headers: {'x-or-provider': 'DeepInfra', 'x-or-generation-id': 'gen-abc123xyz'}
   ```

### Verification Methods

#### Method 1: Use HTTP-Only Mode (Recommended)
For the most detailed logging:
1. Enable **"Use HTTP-only for OpenRouter (bypass SDK)"** toggle
2. This provides full access to response headers including `x-or-provider`
3. Console will show the actual provider that handled your request

#### Method 2: Check Response Model
Even with SDK mode:
- The response model field may contain provider information
- Look for console output: `✅ OpenRouter Response Model: ...`

#### Method 3: Check Saved Payloads
If **SAVE_PAYLOAD** is enabled:
- Check the `openrouter_configs` folder
- Each config JSON file includes:
  - `preferred_provider`: What you requested
  - `model`: The model used
  - `timestamp`: When the request was made

### Example Verification Scenarios

#### Scenario 1: Provider Successfully Changed
```
🔀 OpenRouter: Requesting DeepInfra provider
✅ OpenRouter Provider (from header): DeepInfra
```
**Result**: ✅ Your request was successfully routed to DeepInfra

#### Scenario 2: Provider Fallback
```
🔀 OpenRouter: Requesting DeepInfra provider
✅ OpenRouter Provider (from header): OpenInference
```
**Result**: ⚠️ DeepInfra was unavailable, OpenRouter fell back to OpenInference

#### Scenario 3: Auto Mode
```
📋 OpenRouter Response Model: deepseek/deepseek-chat-v3.1:free
✅ OpenRouter Provider (from header): Together
```
**Result**: ℹ️ Auto mode - OpenRouter chose Together provider automatically

## Technical Details

### OpenRouter API Provider Field
According to OpenRouter's API documentation, you can specify provider preferences using the `provider` field in the request body:

```json
{
  "model": "openrouter/deepseek/deepseek-chat-v3.1:free",
  "messages": [...],
  "provider": {
    "order": ["DeepInfra"],
    "require_parameters": true
  }
}
```

Glossarion now sends this field when a non-"Auto" provider is selected.

### Fallback Behavior
- If "Auto" is selected (default), no provider field is sent, and OpenRouter chooses automatically
- If a specific provider is unavailable, OpenRouter may fall back to another provider
- The console will display `🔀 OpenRouter: Requesting {provider} provider` when a specific provider is requested

## Benefits

1. **Avoids Censorship**: Users can avoid heavily censored providers like Open Inference
2. **Consistency**: Ensures the same provider is used across translation sessions
3. **Flexibility**: Easy to switch providers if one becomes unavailable or performs poorly
4. **Transparency**: Console logging shows which provider was requested
5. **Backward Compatible**: Default "Auto" setting maintains existing behavior

## Related Settings

- **Use HTTP-only for OpenRouter (bypass SDK)**: Forces direct HTTP requests instead of using OpenAI SDK
- **Disable compression for OpenRouter**: Sends `Accept-Encoding: identity` to request uncompressed responses
- **Disable API Safety Filters**: Disables content moderation for supported providers

## Notes

- Not all models are available on all providers
- Provider availability may change over time
- Some providers may have different rate limits or performance characteristics
- The provider preference is a request to OpenRouter, not a guarantee - OpenRouter may route to a different provider if the requested one is unavailable

## Troubleshooting

### Provider Not Changing
If you're still being routed to the wrong provider:

1. **Verify the setting is saved**:
   - Check that the dropdown shows your preferred provider
   - Close and reopen Other Settings to confirm

2. **Enable HTTP-only mode**:
   - Some SDKs may not properly pass the provider field
   - HTTP-only mode guarantees the field is sent

3. **Check console logs**:
   - Look for `🔀 OpenRouter: Requesting {provider}` - confirms request
   - Look for `✅ OpenRouter Provider (from header): {provider}` - confirms response

4. **Provider may be unavailable**:
   - OpenRouter falls back to available providers
   - Try a different provider from the dropdown

### No Logging Output
- Ensure you're running translation (not just opening settings)
- Check that you're using an OpenRouter model
- Console output appears during active translation

### Censorship Issues Persist
- Verify the actual provider used (check logs)
- If still using Open Inference, try:
  - Select "DeepInfra" explicitly
  - Enable HTTP-only mode
  - Contact OpenRouter support about provider availability

## Quick Reference

### Settings Location
**Main Menu** → **Settings** → **Other Settings** → **API Safety Settings** → **Preferred OpenRouter Provider**

### Console Output Reference
| Emoji | Meaning |
|-------|----------|
| 🔀 | Request: Which provider you requested |
| ✅ | Response: Which provider actually handled it |
| 📋 | Model information from response |
| 🆔 | OpenRouter generation ID for support |
| 📊 | All OpenRouter-specific headers |

### Recommended Settings for Issue #36
To avoid Open Inference censorship:
```
Preferred OpenRouter Provider: DeepInfra
Use HTTP-only for OpenRouter: ✓ Enabled (recommended)
Disable compression for OpenRouter: ✗ Disabled (unless needed)
```

### Environment Variables (Advanced)
If modifying directly:
```bash
OPENROUTER_PREFERRED_PROVIDER=DeepInfra
OPENROUTER_USE_HTTP_ONLY=1  # Optional but recommended
```
