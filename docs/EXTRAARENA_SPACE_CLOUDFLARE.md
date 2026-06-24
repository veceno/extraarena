# ExtraArena `.space` Routing Runbook

This document describes the production routing used to keep Telegram on one stable WebApp domain while avoiding the RU origin for non-RU users.

## Current Traffic Model

```text
Telegram WebApp button
  -> https://app.extraarena.space
      -> Cloudflare Redirect Rule
          RU page navigation
            -> 302 https://app.laveqox.ru/<same path>?<same query>
          /ready, /api, /socket.io, non-RU traffic
            -> Cloudflare Tunnel
              -> cloudflared on server
                -> http://localhost:18081
                  -> extraarena-app Docker container
```

`app.laveqox.ru` remains the direct RU entrypoint. Do not put `app.laveqox.ru` behind this Cloudflare Tunnel.

## Domain And DNS

The domain `extraarena.space` is delegated to Cloudflare nameservers:

```text
elisa.ns.cloudflare.com
kolton.ns.cloudflare.com
```

At the registrar, these Cloudflare nameservers replace:

```text
ns1.reg.ru
ns2.reg.ru
```

The game entrypoint is:

```text
app.extraarena.space
```

`app.extraarena.space` is routed through Cloudflare Tunnel. It does not need an `A` or `AAAA` record pointing at the game server.

The effective DNS record is a proxied tunnel route, either created by the Cloudflare Tunnel public hostname UI or represented as:

```text
Type: CNAME
Name: app
Target: <tunnel-id>.cfargotunnel.com
Proxy: Proxied
TTL: Auto
```

The current tunnel ID is:

```text
cfb0bcec-7e0a-4283-aec2-30ebfe4b7f09
```

Authoritative DNS check:

```bash
dig @1.1.1.1 +short extraarena.space NS
dig @1.1.1.1 +short app.extraarena.space A
```

Expected nameservers:

```text
elisa.ns.cloudflare.com.
kolton.ns.cloudflare.com.
```

Expected `app` result is Cloudflare anycast IPs, not the origin server IP.

## Cloudflare Tunnel

Cloudflare Zero Trust tunnel:

```text
Name: extraarena-space
Public hostname: app.extraarena.space
Service type: HTTP
Service URL: localhost:18081
```

The connector runs on the game server via systemd:

```text
Service: cloudflared-extraarena.service
Binary: /home/veceno/bin/cloudflared
Secret env: /home/veceno/extraarena-secrets/cloudflared-extraarena.env
```

The secret env file contains:

```env
TUNNEL_TOKEN=<Cloudflare Tunnel connector token>
```

Do not commit the token. Do not pass it as a process argument. The systemd unit uses `TUNNEL_TOKEN` from the environment so the token does not appear in `ps` or normal `systemctl status` output.

Repository template:

```text
deploy/systemd/cloudflared-extraarena.service
```

Install or update the service:

```bash
sudo install -m 0644 deploy/systemd/cloudflared-extraarena.service /etc/systemd/system/cloudflared-extraarena.service
sudo systemctl daemon-reload
sudo systemctl enable --now cloudflared-extraarena.service
```

Service checks:

```bash
systemctl is-active cloudflared-extraarena.service
systemctl status cloudflared-extraarena.service --no-pager
journalctl -u cloudflared-extraarena.service --since "15 minutes ago" --no-pager
```

Expected state:

```text
active
Registered tunnel connection
Updated to new configuration ... "hostname":"app.extraarena.space","service":"http://localhost:18081"
```

## Cloudflare Redirect Rule

The active production redirect is a Cloudflare **Redirect Rule**, not a Worker.

Location in Cloudflare:

```text
extraarena.space -> Rules -> Redirect Rules
```

Rule name:

```text
RuRedirect
```

Use **Custom filter expression**:

```txt
(http.host eq "app.extraarena.space"
 and ip.src.country eq "RU"
 and not starts_with(http.request.uri.path, "/ready")
 and not starts_with(http.request.uri.path, "/api")
 and not starts_with(http.request.uri.path, "/socket.io"))
```

Redirect settings:

```text
Type: Dynamic
Expression: concat("https://app.laveqox.ru", http.request.uri.path)
Status code: 302
Preserve query string: enabled
```

Why these exclusions exist:

- `/ready` must stay on `.space` so health checks test the tunnel honestly.
- `/api` must stay same-origin for the loaded app and avoid cross-origin auth/socket surprises.
- `/socket.io` must stay on the same host as the page runtime using it.

Keep the status code as `302`. Do not use `301` while Telegram WebView behavior and regional routing may still change.

## Optional Worker Fallback

`deploy/cloudflare/app-extraarena-space-worker.js` contains an equivalent Worker implementation. It is not the active production path right now.

Use it only if Redirect Rules become insufficient. The Cloudflare dashboard uploader may reject Worker projects when JavaScript files are detected; in that case use `wrangler deploy`.

## Application Environment

The Docker runtime reads production environment from:

```text
/home/veceno/extraarena-secrets/.env.production
```

Relevant production values:

```env
WEBAPP_URL=https://app.extraarena.space
EXTRA_SHOP_URL=https://app.extraarena.space
CORS_ALLOWED_ORIGINS=https://app.extraarena.space,https://app.laveqox.ru
MCP_ALLOWED_ORIGINS=https://app.extraarena.space,https://app.laveqox.ru

ROBOKASSA_RESULT_URL=https://app.extraarena.space/api/payments/robokassa/result
ROBOKASSA_SUCCESS_URL=https://app.extraarena.space/extraShop/payment-success
ROBOKASSA_FAIL_URL=https://app.extraarena.space/extraShop/payment-fail
```

Why Robokassa URLs use `.space`: non-RU users should not be returned to the RU-only domain after checkout.

Local example values are documented in `.env.example`.

## Docker Runtime

The live game container is managed by Docker Compose:

```text
Compose project: extraarena
Working dir: /mnt/veceno1/extraarena/project/current
Compose file: /mnt/veceno1/extraarena/project/current/compose.yaml
Container: extraarena-app
Network mode: host
Internal web port: 18081
```

MCP-uploaded profile cosmetics are runtime data, not image data. Keep the
admin upload directories on the shared disk and bind-mount them into the
container on every rebuild:

```yaml
services:
  app:
    volumes:
      - /mnt/veceno1/extraarena/project/shared/DesignAssets/PlayerCosmetics/Avatars/Admin:/app/DesignAssets/PlayerCosmetics/Avatars/Admin
      - /mnt/veceno1/extraarena/project/shared/DesignAssets/PlayerCosmetics/Background/Admin:/app/DesignAssets/PlayerCosmetics/Background/Admin
```

Do not rsync or rebuild these `Admin` directories as part of the application
image, and do not run deploy cleanup with `--delete` against the shared
`DesignAssets/PlayerCosmetics/*/Admin` paths. Source-controlled starter
cosmetics may still be copied into the image; the bind mounts overlay only the
MCP-managed upload directories at runtime.

Restart the app after environment changes:

```bash
cd /mnt/veceno1/extraarena/project/current
sudo docker compose up -d --force-recreate app
```

Container checks:

```bash
sudo docker ps --filter name=extraarena-app
sudo docker inspect -f '{{.State.Health.Status}}' extraarena-app
sudo docker logs --tail 120 extraarena-app
```

MCP cosmetics preservation check after a rebuild:

```bash
sudo docker compose exec app find /app/DesignAssets/PlayerCosmetics/Avatars/Admin -maxdepth 1 -type f | wc -l
sudo docker compose exec app find /app/DesignAssets/PlayerCosmetics/Background/Admin -maxdepth 1 -type f | wc -l
curl -I https://app.extraarena.space/DesignAssets/PlayerCosmetics/Avatars/Admin/<known-file>
```

Verify the running environment:

```bash
pid="$(pgrep -f '^python main.py$')"
sudo tr '\0' '\n' < "/proc/$pid/environ" \
  | grep -E '^(WEBAPP_URL|EXTRA_SHOP_URL|CORS_ALLOWED_ORIGINS|MCP_ALLOWED_ORIGINS|ROBOKASSA_(RESULT|SUCCESS|FAIL)_URL)='
```

Expected log line after restart:

```text
Bot started in production. WebApp: https://app.extraarena.space
```

## Telegram

BotFather / Telegram Mini App domain should be:

```text
app.extraarena.space
```

The bot code uses `WEBAPP_URL` to create `WebAppInfo` buttons, so after the container restart new Telegram buttons point at `.space`.

## Verification

DNS:

```bash
dig @1.1.1.1 +short extraarena.space NS
dig @1.1.1.1 +short app.extraarena.space A
```

Tunnel health:

```bash
curl -i https://app.extraarena.space/ready
```

Expected:

```text
HTTP/2 200
{"status": "ok", "service": "extraarena-webapp", ...}
```

Redirect rule from a RU network:

```bash
curl -I 'https://app.extraarena.space/?x=1'
```

Expected:

```text
HTTP/2 302
location: https://app.laveqox.ru/?x=1
```

Bypass checks from a RU network:

```bash
curl -I https://app.extraarena.space/ready
curl -I https://app.extraarena.space/api/config
```

Expected:

- `/ready` returns `200`, not `302`.
- `/api/...` is served by the app, not redirected. A `404` for an unknown API path is acceptable; a `302` is not.

Direct RU entrypoint:

```bash
curl -i https://app.laveqox.ru/ready
```

Expected:

```text
HTTP/1.1 200 OK
```

Telegram smoke:

1. Open the bot button.
2. Confirm profile loads with Telegram init data.
3. Start a bot battle.
4. Confirm Socket.IO connects in arena.
5. Open shop/payment flow if payments are enabled.

## Rollback

Fast rollback to the previous direct RU domain:

1. In Cloudflare, disable the Redirect Rule.
2. In BotFather, set the WebApp domain back to:

   ```text
   app.laveqox.ru
   ```

3. On the server, restore production URL values in `/home/veceno/extraarena-secrets/.env.production`:

   ```env
   WEBAPP_URL=https://app.laveqox.ru
   EXTRA_SHOP_URL=https://app.laveqox.ru
   CORS_ALLOWED_ORIGINS=https://app.laveqox.ru
   MCP_ALLOWED_ORIGINS=https://app.laveqox.ru
   ROBOKASSA_RESULT_URL=https://app.laveqox.ru/api/payments/robokassa/result
   ROBOKASSA_SUCCESS_URL=https://app.laveqox.ru/extraShop/payment-success
   ROBOKASSA_FAIL_URL=https://app.laveqox.ru/extraShop/payment-fail
   ```

4. Recreate the container:

   ```bash
   cd /mnt/veceno1/extraarena/project/current
   sudo docker compose up -d --force-recreate app
   ```

5. Optional: stop the tunnel if not needed:

   ```bash
   sudo systemctl disable --now cloudflared-extraarena.service
   ```

## Mobile App (Android)

The Cloudflare `RuRedirect` rule only rewrites **page navigations** for RU IPs. The Android app
serves its page HTML/JS/CSS from the APK (`shouldInterceptRequest`), so the edge redirect never
fires for it — `/api` and `/socket.io` go to whatever host the app selects. The app therefore
selects the host itself:

- Two built-in connection profiles: `extraarena_worldwide` → `app.extraarena.space`,
  `extraarena_ru` → `app.laveqox.ru`.
- On first run (no prior selection), `RegionDetector` picks by device region (SIM/network country
  ISO, locale, timezone — no permission required): RU → `app.laveqox.ru`, else →
  `app.extraarena.space`. A pre-existing or manual (hidden-switcher) selection is respected.
- If the selected host's `/health` probe fails, the app falls back to the other built-in host.

`/api` and `/socket.io` always stay same-origin with the selected host. Both hosts are already in
`CORS_ALLOWED_ORIGINS`. See `android-app/docs/ARCHITECTURE.md` for the full contract.

## Files In This Repository

```text
docs/EXTRAARENA_SPACE_CLOUDFLARE.md
deploy/systemd/cloudflared-extraarena.service
deploy/cloudflared/extraarena-space.yml
deploy/cloudflare/app-extraarena-space-worker.js
.env.example
start.sh
```

`deploy/cloudflared/extraarena-space.yml` is useful for locally-managed tunnels. The active server setup currently uses a remote-managed token-based tunnel through systemd.
