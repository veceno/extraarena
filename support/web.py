from __future__ import annotations

import base64
import hmac
import json
import time
from hashlib import sha256
from typing import Any

from aiohttp import web

from support.attachments import (
    SUPPORT_UPLOAD_ROOT,
    compress_support_attachment,
    resolve_support_attachment_path,
)
from support.bot_client import SupportBotConversationManager, format_text_menu
from support.codes import generate_support_code
from support.constants import AccountScope, SupportChannel, SupportTopic
from support.channels.max import extract_max_image_attachments, normalize_max_message, verify_max_webhook_secret


SUPPORT_ADMIN_COOKIE = "ea_support_admin"
SUPPORT_ADMIN_COOKIE_MAX_AGE_SECONDS = 4 * 60 * 60
ADMIN_LOGIN_REQUEST_WINDOW_SECONDS = 60
ADMIN_LOGIN_REQUEST_MAX = 3
PUBLIC_TICKET_WINDOW_SECONDS = 60
PUBLIC_TICKET_MAX = 10
MAX_SUPPORT_BODY_CHARS = 4000


SUPPORT_PAGE_HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ExtraArena Support</title></head>
<body><main><h1>ExtraArena Support</h1><p>Выберите тему, укажите аккаунт и опишите проблему.</p></main></body></html>
"""


SUPPORT_ADMIN_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ExtraArena Support Admin</title>
<style>
:root{color-scheme:dark;--bg:#101214;--panel:#171a1e;--line:#2a3037;--text:#edf1f5;--muted:#9aa6b2;--accent:#3dd6a2;--warn:#f3bc5f;--danger:#ff7d7d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
button,input,textarea,select{font:inherit}button{border:1px solid var(--line);background:#20262d;color:var(--text);border-radius:6px;padding:8px 10px;cursor:pointer}button:hover{border-color:#46515c}
.app{min-height:100vh;display:grid;grid-template-rows:auto 1fr}.topbar{height:52px;display:flex;align-items:center;justify-content:space-between;padding:0 16px;border-bottom:1px solid var(--line);background:#121519}
.brand{font-weight:700}.muted{color:var(--muted)}.login{max-width:420px;margin:12vh auto;padding:24px;border:1px solid var(--line);background:var(--panel);border-radius:8px}.login h1{margin:0 0 12px;font-size:22px}
.login input{width:100%;margin:10px 0;padding:10px;border-radius:6px;border:1px solid var(--line);background:#0d0f12;color:var(--text)}
.workspace{display:grid;grid-template-columns:minmax(280px,360px) 1fr;min-height:0}.inbox{border-right:1px solid var(--line);background:#121519;overflow:auto}.inbox-head{display:flex;gap:8px;align-items:center;padding:12px;border-bottom:1px solid var(--line);position:sticky;top:0;background:#121519}
.ticket{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid var(--line);border-radius:0;background:transparent;padding:12px}.ticket.active{background:#1c2429}.ticket-title{display:flex;gap:8px;align-items:center;justify-content:space-between}.pill{font-size:12px;border:1px solid var(--line);border-radius:999px;padding:2px 7px;color:var(--muted)}
.pill.ultra,.pill.extra_pass{color:#0d1512;background:var(--accent);border-color:var(--accent)}.pill.guest{color:var(--muted)}.snippet{margin-top:6px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.detail{display:grid;grid-template-rows:auto 1fr auto;min-width:0}.detail-head{padding:14px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:12px}.meta{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
.messages{padding:16px;overflow:auto;display:flex;flex-direction:column;gap:10px}.msg{max-width:760px;border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--panel)}.msg.outbound{margin-left:auto;background:#182520}.msg.internal{background:#282119;border-color:#51402a}.msg .time{font-size:12px;color:var(--muted);margin-top:6px}
.attachments{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}.attachments img{max-width:160px;max-height:120px;border-radius:6px;border:1px solid var(--line);object-fit:cover}
.composer{border-top:1px solid var(--line);padding:12px 16px;background:#121519}.composer textarea{width:100%;min-height:84px;resize:vertical;border-radius:6px;border:1px solid var(--line);background:#0d0f12;color:var(--text);padding:10px}
.composer-actions{margin-top:8px;display:flex;gap:8px;align-items:center;justify-content:space-between}.status-select{background:#0d0f12;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:8px}.empty{display:grid;place-items:center;color:var(--muted);height:100%}
@media(max-width:760px){.workspace{grid-template-columns:1fr}.inbox{max-height:42vh;border-right:0;border-bottom:1px solid var(--line)}.detail{min-height:58vh}}
</style>
</head>
<body>
<div id="app" class="app">
  <div class="topbar"><div class="brand">ExtraArena Support</div><div id="sessionState" class="muted">Проверка сессии...</div></div>
  <main id="root"></main>
</div>
<script>
const root=document.getElementById('root');
const sessionState=document.getElementById('sessionState');
let tickets=[];let selectedId=null;let selectedDetail=null;let mode='reply';
const statuses=['open','queued_unverified','queued_guest','pending_admin','closed'];
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function api(path,opts={}){const r=await fetch(path,{credentials:'same-origin',headers:{'Content-Type':'application/json',...(opts.headers||{})},...opts});let data={};try{data=await r.json();}catch(e){}if(!r.ok)throw Object.assign(new Error(data.error||'request_failed'),{status:r.status,data});return data;}
function renderLogin(msg=''){sessionState.textContent='Нужен вход';root.innerHTML=`<section class="login"><h1>Support Admin</h1><p class="muted">Код придет в support Telegram-бот.</p><button id="requestCode">Получить код</button><input id="code" autocomplete="one-time-code" placeholder="Одноразовый код"><button id="verifyCode">Войти</button><p class="muted">${esc(msg)}</p></section>`;document.getElementById('requestCode').onclick=async()=>{await api('/api/support/admin/login/request',{method:'POST',body:'{}'});renderLogin('Код отправлен.');};document.getElementById('verifyCode').onclick=async()=>{const code=document.getElementById('code').value.trim();await api('/api/support/admin/login/verify',{method:'POST',body:JSON.stringify({code})});await loadTickets();};}
async function loadTickets(){try{const data=await api('/api/support/admin/tickets');tickets=data.tickets||[];sessionState.textContent=`${tickets.length} обращений`;if(!selectedId&&tickets[0])selectedId=tickets[0].id;renderWorkspace();if(selectedId)await loadDetail(selectedId);}catch(e){if(e.status===401)renderLogin();else{sessionState.textContent='Ошибка';root.innerHTML=`<div class="empty">${esc(e.message)}</div>`;}}}
async function loadDetail(id){selectedId=id;const data=await api(`/api/support/admin/tickets/${encodeURIComponent(id)}`);selectedDetail=data;renderWorkspace();}
function renderWorkspace(){root.innerHTML=`<section class="workspace"><aside class="inbox"><div class="inbox-head"><strong>Inbox</strong><button id="refresh">Обновить</button></div>${tickets.map(t=>`<button class="ticket ${t.id===selectedId?'active':''}" data-id="${esc(t.id)}"><div class="ticket-title"><strong>${esc(t.subject||t.topic||'Без темы')}</strong><span class="pill ${esc(t.priority_tier)}">${esc(t.priority_tier||'')}</span></div><div class="meta"><span class="pill">${esc(t.channel||'')}</span><span class="pill">${esc(t.status||'')}</span><span class="pill">${esc(t.account_scope||'')}</span></div><div class="snippet">${esc(t.latest_message_body||'')}</div></button>`).join('')}</aside><section class="detail">${renderDetail()}</section></section>`;document.getElementById('refresh').onclick=loadTickets;document.querySelectorAll('.ticket').forEach(btn=>btn.onclick=()=>loadDetail(btn.dataset.id));bindDetail();}
function renderDetail(){if(!selectedDetail)return '<div class="empty">Выберите обращение</div>';const t=selectedDetail.ticket||{};const messages=selectedDetail.messages||[];const attachments=selectedDetail.attachments||[];return `<header class="detail-head"><div><strong>${esc(t.subject||t.topic||'Обращение')}</strong><div class="meta"><span class="pill">${esc(t.id)}</span><span class="pill">${esc(t.channel)}:${esc(t.channel_id)}</span><span class="pill ${esc(t.priority_tier)}">${esc(t.priority_tier)}</span></div></div><select id="status" class="status-select">${statuses.map(s=>`<option value="${s}" ${s===t.status?'selected':''}>${s}</option>`).join('')}</select></header><div class="messages">${messages.map(m=>renderMessage(m,attachments)).join('')}</div><footer class="composer"><textarea id="body" placeholder="Ответ пользователю или внутренняя заметка"></textarea><div class="composer-actions"><div><button id="reply">Ответить</button><button id="note">Заметка</button></div><span class="muted" id="actionState"></span></div></footer>`;}
function renderMessage(m,attachments){const linked=attachments.filter(a=>String(a.message_id||'')===String(m.id));return `<article class="msg ${esc(m.direction)}"><div>${esc(m.body)}</div>${linked.length?`<div class="attachments">${linked.map(a=>`<a href="${esc(a.storage_path)}" target="_blank" rel="noopener"><img src="${esc(a.storage_path)}" alt="${esc(a.original_filename||'attachment')}"></a>`).join('')}</div>`:''}<div class="time">${esc(m.direction)} · ${esc(m.created_at||'')}</div></article>`;}
function bindDetail(){if(!selectedDetail)return;const status=document.getElementById('status');const body=document.getElementById('body');const state=document.getElementById('actionState');if(status)status.onchange=async()=>{await api(`/api/support/admin/tickets/${encodeURIComponent(selectedId)}/status`,{method:'POST',body:JSON.stringify({status:status.value})});await loadTickets();};async function send(kind){const text=body.value.trim();if(!text)return;const path=kind==='note'?'note':'reply';state.textContent='Отправка...';await api(`/api/support/admin/tickets/${encodeURIComponent(selectedId)}/${path}`,{method:'POST',body:JSON.stringify({body:text})});body.value='';state.textContent='Готово';await loadDetail(selectedId);await loadTickets();}const reply=document.getElementById('reply');const note=document.getElementById('note');if(reply)reply.onclick=()=>send('reply');if(note)note.onclick=()=>send('note');}
loadTickets();
</script>
</body>
</html>
"""


def _json_response(payload: dict[str, Any], *, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status)


def _sign_admin_session(admin_channel_id: str, secret: str, *, now: int | None = None) -> str:
    issued = int(now or time.time())
    payload = {
        "admin_channel_id": str(admin_channel_id),
        "iat": issued,
        "exp": issued + SUPPORT_ADMIN_COOKIE_MAX_AGE_SECONDS,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    signature = hmac.new(secret.encode(), encoded.encode(), sha256).hexdigest()
    return f"{encoded}.{signature}"


def _ticket_access_token(ticket_id: str, secret: str) -> str:
    return hmac.new(secret.encode(), str(ticket_id).encode(), sha256).hexdigest()


def _verify_ticket_access(ticket_id: str, token: str | None, secret: str) -> bool:
    if not token:
        return False
    return hmac.compare_digest(_ticket_access_token(ticket_id, secret), str(token))


def _bearer_token(request: web.Request) -> str:
    authorization = str(request.headers.get("Authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return str(request.rel_url.query.get("access_token") or "").strip()


def _verify_admin_session(token: str, secret: str) -> dict[str, Any] | None:
    try:
        encoded, signature = str(token or "").split(".", 1)
    except ValueError:
        return None
    expected = hmac.new(secret.encode(), encoded.encode(), sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return None
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
    except Exception:
        return None
    if int(payload.get("exp") or 0) < int(time.time()):
        return None
    return payload


async def require_support_admin(request: web.Request, admin_secret: str) -> dict[str, Any]:
    payload = _verify_admin_session(str(request.cookies.get(SUPPORT_ADMIN_COOKIE) or ""), admin_secret)
    if not payload:
        raise web.HTTPUnauthorized(
            text=json.dumps({"error": "support_admin_auth_required"}),
            content_type="application/json",
        )
    return payload


def register_support_routes(
    app: web.Application,
    support_service: Any,
    *,
    admin_secret: str,
    admin_channel_id: str = "",
    max_webhook_secret: str = "",
    upload_root=SUPPORT_UPLOAD_ROOT,
) -> None:
    admin_login_attempts: dict[str, list[float]] = {}
    public_ticket_attempts: dict[str, list[float]] = {}
    max_conversation = SupportBotConversationManager(support_service)

    async def support_page(request: web.Request) -> web.Response:
        return web.Response(text=SUPPORT_PAGE_HTML, content_type="text/html")

    async def support_admin_page(request: web.Request) -> web.Response:
        return web.Response(text=SUPPORT_ADMIN_HTML, content_type="text/html")

    async def create_ticket(request: web.Request) -> web.Response:
        remote = str(request.remote or "unknown")
        now = time.time()
        attempts = [ts for ts in public_ticket_attempts.get(remote, []) if now - ts < PUBLIC_TICKET_WINDOW_SECONDS]
        if len(attempts) >= PUBLIC_TICKET_MAX:
            return _json_response({"error": "rate_limit_exceeded"}, status=429)
        attempts.append(now)
        public_ticket_attempts[remote] = attempts
        try:
            data = await request.json()
        except Exception:
            return _json_response({"error": "invalid_json"}, status=400)
        topic = str(data.get("topic") or SupportTopic.OTHER).strip()
        if topic not in SupportTopic.ALL:
            return _json_response({"error": "invalid_topic"}, status=400)
        body = str(data.get("body") or "").strip()
        if not body:
            return _json_response({"error": "body_required"}, status=400)
        if len(body) > MAX_SUPPORT_BODY_CHARS:
            return _json_response({"error": "body_too_large"}, status=400)
        is_guest = bool(data.get("guest") or data.get("cannot_get_code"))
        game_user_id = data.get("game_user_id")
        try:
            game_user_id = int(game_user_id) if game_user_id not in (None, "") else None
        except (TypeError, ValueError):
            return _json_response({"error": "invalid_game_user_id"}, status=400)
        trusted_game_user_id = None
        account_scope = AccountScope.GUEST if is_guest else AccountScope.UNVERIFIED
        result = await support_service.create_ticket(
            topic=topic,
            channel=SupportChannel.SITE,
            channel_id=str(data.get("channel_id") or data.get("session_id") or "site"),
            external_user_id=str(data.get("external_user_id") or ""),
            display_name=str(data.get("display_name") or "Guest"),
            body=body,
            game_user_id=trusted_game_user_id,
            account_scope=account_scope,
            subject=str(data.get("subject") or ""),
            metadata={"source": "site", "claimed_game_user_id": game_user_id},
        )
        ticket = result.get("ticket") or {}
        access_token = _ticket_access_token(str(ticket.get("id") or ""), admin_secret) if ticket.get("id") else ""
        return _json_response({
            "status": "ok",
            "ticket": ticket,
            "message": result.get("message"),
            "ticket_access_token": access_token,
        })

    async def upload_attachment(request: web.Request) -> web.Response:
        try:
            reader = await request.multipart()
            field = await reader.next()
            form_fields: dict[str, str] = {}
            while field is not None and field.name != "file":
                form_fields[field.name] = (await field.read(decode=True)).decode("utf-8", errors="ignore")
                field = await reader.next()
            if field is None or field.name != "file":
                return _json_response({"error": "file_field_required"}, status=400)
            ticket_id = str(form_fields.get("ticket_id") or "").strip()
            if not ticket_id:
                return _json_response({"error": "ticket_id_required"}, status=400)
            admin_payload = _verify_admin_session(str(request.cookies.get(SUPPORT_ADMIN_COOKIE) or ""), admin_secret)
            ticket_token = form_fields.get("access_token")
            if not admin_payload and not _verify_ticket_access(ticket_id, ticket_token, admin_secret):
                return _json_response({"error": "ticket_access_required"}, status=401)
            content_type = field.headers.get("Content-Type", "")
            if content_type not in {"image/png", "image/jpeg", "image/webp"}:
                return _json_response({"error": "invalid_content_type"}, status=400)
            data = await field.read(decode=True)
            metadata = compress_support_attachment(
                data,
                original_filename=field.filename,
                upload_root=upload_root,
            )
            try:
                record = await support_service.record_attachment(
                    ticket_id=ticket_id,
                    message_id=(form_fields.get("message_id") or None) if admin_payload else None,
                    uploader_identity_id=(form_fields.get("identity_id") or None) if admin_payload else None,
                    metadata=metadata.as_record(upload_root=upload_root),
                )
            except Exception:
                try:
                    metadata.path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            return _json_response({"status": "ok", "attachment": record})
        except ValueError as exc:
            return _json_response({"error": str(exc)}, status=400)
        except Exception:
            return _json_response({"error": "attachment_upload_failed"}, status=500)

    async def ticket_messages(request: web.Request) -> web.Response:
        ticket_id = request.match_info["ticket_id"]
        if not _verify_ticket_access(ticket_id, _bearer_token(request), admin_secret):
            return _json_response({"error": "ticket_access_required"}, status=401)
        messages = await support_service.list_ticket_messages(ticket_id=request.match_info["ticket_id"], public=True)
        return _json_response({"status": "ok", "messages": messages})

    async def support_attachment_static(request: web.Request) -> web.StreamResponse:
        try:
            path = resolve_support_attachment_path(
                f"{request.match_info['year']}/{request.match_info['month']}/{request.match_info['filename']}",
                upload_root=upload_root,
            )
        except ValueError:
            raise web.HTTPForbidden()
        if not path.exists() or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    async def max_webhook(request: web.Request) -> web.Response:
        if not verify_max_webhook_secret(max_webhook_secret, request.headers.get("X-Max-Bot-Api-Secret")):
            return _json_response({"error": "invalid_max_webhook_secret"}, status=401)
        try:
            data = await request.json()
        except Exception:
            return _json_response({"error": "invalid_json"}, status=400)
        message = normalize_max_message(data)
        max_attachments = extract_max_image_attachments(data)
        if not message.get("body") and not max_attachments:
            return _json_response({"status": "ignored"})
        body = message["body"] or "Вложение"
        reply = await max_conversation.receive_text(
            channel=SupportChannel.MAX,
            channel_id=message["channel_id"],
            external_user_id=message["external_user_id"],
            display_name=message["display_name"],
            body=body,
            metadata={"raw": message["raw"], "has_attachment": bool(max_attachments)},
        )
        rendered_reply = format_text_menu(reply)
        max_client = request.app.get("support_max_client")
        saved_attachments = 0
        if max_client and reply.ticket and reply.message:
            for attachment in max_attachments:
                try:
                    downloaded = await max_client.download_url(attachment["url"])
                    metadata = compress_support_attachment(
                        downloaded,
                        original_filename=attachment.get("filename") or "max-image",
                        upload_root=upload_root,
                    )
                    await support_service.record_attachment(
                        ticket_id=reply.ticket["id"],
                        message_id=reply.message["id"],
                        uploader_identity_id=(reply.identity or {}).get("id") or reply.ticket.get("requester_identity_id"),
                        metadata=metadata.as_record(upload_root=upload_root),
                    )
                    saved_attachments += 1
                except ValueError:
                    continue
        if max_attachments and not saved_attachments:
            rendered_reply += "\n\nВложение пока не сохранено: завершите создание обращения или отправьте изображение еще раз."
        if max_client and message.get("channel_id"):
            await max_client.send_message(message["channel_id"], rendered_reply)
        return _json_response({"status": "ok", "reply": rendered_reply, "attachments_saved": saved_attachments})

    async def admin_login_request(request: web.Request) -> web.Response:
        remote = str(request.remote or "unknown")
        now = time.time()
        attempts = [ts for ts in admin_login_attempts.get(remote, []) if now - ts < ADMIN_LOGIN_REQUEST_WINDOW_SECONDS]
        if len(attempts) >= ADMIN_LOGIN_REQUEST_MAX:
            return _json_response({"error": "rate_limit_exceeded"}, status=429)
        attempts.append(now)
        admin_login_attempts[remote] = attempts
        code = generate_support_code()
        await support_service.issue_admin_login_code(
            admin_channel_id=str(admin_channel_id or "admin"),
            code=code,
            ttl_seconds=300,
            metadata={"source": "support_admin"},
        )
        notifier = request.app.get("support_admin_notifier")
        if notifier and hasattr(notifier, "send_admin_code"):
            await notifier.send_admin_code(code)
        return _json_response({"status": "ok"})

    async def admin_login_verify(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return _json_response({"error": "invalid_json"}, status=400)
        code = str(data.get("code") or "").strip()
        if not code:
            return _json_response({"error": "code_required"}, status=400)
        consumed = await support_service.consume_admin_login_code(code)
        if not consumed:
            return _json_response({"error": "invalid_or_expired_code"}, status=401)
        channel_id = str(consumed.get("admin_channel_id") or admin_channel_id or "admin")
        response = _json_response({"status": "ok"})
        response.set_cookie(
            SUPPORT_ADMIN_COOKIE,
            _sign_admin_session(channel_id, admin_secret),
            max_age=SUPPORT_ADMIN_COOKIE_MAX_AGE_SECONDS,
            httponly=True,
            secure=bool(request.secure or str(request.headers.get("X-Forwarded-Proto", "")).lower() == "https"),
            samesite="Lax",
            path="/",
        )
        return response

    async def admin_tickets(request: web.Request) -> web.Response:
        await require_support_admin(request, admin_secret)
        tickets = await support_service.list_admin_tickets()
        return _json_response({"status": "ok", "tickets": tickets})

    async def admin_ticket_detail(request: web.Request) -> web.Response:
        await require_support_admin(request, admin_secret)
        ticket_id = request.match_info["ticket_id"]
        ticket = await support_service.get_ticket(ticket_id=ticket_id)
        if not ticket:
            return _json_response({"error": "ticket_not_found"}, status=404)
        messages = await support_service.list_ticket_messages(ticket_id=ticket_id, public=False)
        attachments = await support_service.list_ticket_attachments(ticket_id=ticket_id)
        return _json_response({
            "status": "ok",
            "ticket": ticket,
            "messages": messages,
            "attachments": attachments,
        })

    def _find_ticket(ticket_id: str, tickets: list[dict[str, Any]]) -> dict[str, Any] | None:
        for ticket in tickets:
            if str(ticket.get("id")) == str(ticket_id):
                return ticket
        return None

    async def admin_reply(request: web.Request) -> web.Response:
        admin = await require_support_admin(request, admin_secret)
        try:
            data = await request.json()
        except Exception:
            return _json_response({"error": "invalid_json"}, status=400)
        body = str(data.get("body") or "").strip()
        if not body:
            return _json_response({"error": "body_required"}, status=400)
        ticket_id = request.match_info["ticket_id"]
        ticket = await support_service.get_ticket(ticket_id=ticket_id)
        if ticket is None:
            ticket = _find_ticket(ticket_id, await support_service.list_admin_tickets())
        if not ticket:
            return _json_response({"error": "ticket_not_found"}, status=404)
        result = await support_service.create_admin_reply(
            ticket=ticket,
            body=body,
            admin_channel_id=admin.get("admin_channel_id") or admin_channel_id or "admin",
        )
        return _json_response({"status": "ok", **result})

    async def admin_note(request: web.Request) -> web.Response:
        admin = await require_support_admin(request, admin_secret)
        try:
            data = await request.json()
        except Exception:
            return _json_response({"error": "invalid_json"}, status=400)
        body = str(data.get("body") or "").strip()
        if not body:
            return _json_response({"error": "body_required"}, status=400)
        note = await support_service.create_admin_note(
            ticket_id=request.match_info["ticket_id"],
            body=body,
            admin_channel_id=admin.get("admin_channel_id") or admin_channel_id or "admin",
        )
        return _json_response({"status": "ok", "note": note})

    async def admin_status(request: web.Request) -> web.Response:
        admin = await require_support_admin(request, admin_secret)
        try:
            data = await request.json()
        except Exception:
            return _json_response({"error": "invalid_json"}, status=400)
        status = str(data.get("status") or "").strip()
        if not status:
            return _json_response({"error": "status_required"}, status=400)
        ticket = await support_service.update_ticket_status(
            ticket_id=request.match_info["ticket_id"],
            status=status,
            admin_channel_id=admin.get("admin_channel_id") or admin_channel_id or "admin",
        )
        return _json_response({"status": "ok", "ticket": ticket})

    app.router.add_get("/support", support_page)
    app.router.add_get("/support/", support_page)
    app.router.add_get("/support/admin", support_admin_page)
    app.router.add_get("/support/admin/", support_admin_page)
    app.router.add_post("/api/support/tickets", create_ticket)
    app.router.add_get("/api/support/tickets/{ticket_id}/messages", ticket_messages)
    app.router.add_post("/api/support/attachments", upload_attachment)
    app.router.add_post("/api/support/max/webhook", max_webhook)
    app.router.add_post("/api/support/admin/login/request", admin_login_request)
    app.router.add_post("/api/support/admin/login/verify", admin_login_verify)
    app.router.add_get("/api/support/admin/tickets", admin_tickets)
    app.router.add_get("/api/support/admin/tickets/{ticket_id}", admin_ticket_detail)
    app.router.add_post("/api/support/admin/tickets/{ticket_id}/reply", admin_reply)
    app.router.add_post("/api/support/admin/tickets/{ticket_id}/note", admin_note)
    app.router.add_post("/api/support/admin/tickets/{ticket_id}/status", admin_status)
    app.router.add_get("/uploads/support/{year}/{month}/{filename}", support_attachment_static)
