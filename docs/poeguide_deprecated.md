# POE Authentication Guide for Glossarion

## ⚠️ Important: POE Does Not Use Traditional API Keys

Unlike other AI providers (OpenAI, Anthropic, Google), **POE requires cookie-based authentication** instead of API keys. This is because POE doesn't offer an official API - we're essentially mimicking a browser session.

### Why Cookie Authentication?

- POE uses **HttpOnly cookies** for security
- These cookies cannot be accessed via JavaScript
- You must manually extract them from your browser
- Cookies expire periodically and need to be refreshed

## 📋 How to Get Your POE Cookie

### Quick Reference Table

| Step | Chrome/Edge | Firefox | Safari |
|------|-------------|---------|--------|
| 1. | Go to [poe.com](https://poe.com) and **log in** | Same | Same |
| 2. | Press `F12` to open DevTools | Press `F12` | Press `Cmd+Option+I` |
| 3. | Click **Application** tab | Click **Storage** tab | Click **Storage** tab |
| 4. | Navigate to:<br>`Storage → Cookies → https://poe.com` | Navigate to:<br>`Cookies → https://poe.com` | Navigate to:<br>`Cookies → poe.com` |
| 5. | Find cookie named **`p-b`** | Same | Same |
| 6. | Double-click the **Value** column | Same | Same |
| 7. | Copy the entire value (`Ctrl+C`) | Copy the entire value (`Ctrl+C`) | Copy the entire value (`Cmd+C`) |
| 8. | In Glossarion, paste as:<br>`p-b:YOUR_COPIED_VALUE` | Same | Same |

### Detailed Instructions

1. **Login First**: You MUST be logged into poe.com before attempting to get the cookie
2. **Find the Right Cookie**: Look for `p-b` - ignore other cookies like `p-lat`
3. **Copy the Value Only**: Don't copy the cookie name, just its value
4. **Format Correctly**: In Glossarion's API key field, enter it as `p-b:` followed by the value

### Example Format

p-b:XXXXXXXXXXXXXXXXXXXXXXXX%3D%3D

## 🔄 Cookie Expiration

POE cookies expire after a few days. When you see authentication errors:

1. Return to poe.com and check if you're still logged in
2. Get a fresh cookie following the steps above
3. Update your API key in Glossarion

## 💡 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cookie not found" error | Make sure you're logged into poe.com first |
| Multiple `p-b` cookies | Use the one with the longest value |
| Authentication fails immediately | Cookie may be expired - get a fresh one |
| Can't find Application/Storage tab | Try right-clicking anywhere → Inspect Element |

## ❓ FAQ

**Q: Why can't I just use an API key like other services?**  
A: POE API key doesn't work for some reason. This integration works by mimicking a browser session, which requires authentication cookies.

**Q: How often do I need to update the cookie?**  
A: Typically every 3-7 days, or whenever you see authentication errors.

**Q: Is this safe?**  
A: I hope, please treat your cookie like a password. Anyone with your cookie can access your POE account.

**Q: Can this be automated?**  
A: Not that i know of. POE uses HttpOnly cookies specifically to prevent programmatic access for security reasons.

---

*Note: This method is a workaround since the POE API key is meant for bots only*

*Warning: your POE account can get banned if you abuse this. Please increase your API call delay to reduce this risk.*
